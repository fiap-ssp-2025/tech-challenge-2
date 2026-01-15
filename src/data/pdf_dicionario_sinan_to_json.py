"""
Extrai as tabelas do PDF oficial do SINAN (Dicionário de Dados) e salva em JSON.

Objetivo: converter as tabelas que possuem as colunas
  Nome do Campo | Campo | Tipo | Categoria | Descrição | Características | DBF

Saída: lista de objetos (um por linha da tabela), com exatamente esses 7 atributos.

Uso:
  python3 src/data/pdf_dicionario_sinan_to_json.py \
    --pdf docs/dicionario_dados_sinan.pdf \
    --out src/data/dicionario_dados_sinan_tabelas.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pdfplumber


EXPECTED_COLUMNS = [
    "Nome do Campo",
    "Campo",
    "Tipo",
    "Categoria",
    "Descrição",
    "Características",
    "DBF",
]


def _norm(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _clean_cell(s: Optional[str]) -> str:
    if not s:
        return ""
    # pdfplumber tende a inserir quebras em células longas
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def _pad_or_merge_row(row: List[str], target_len: int = 7) -> List[str]:
    row = [c if c is not None else "" for c in row]
    if len(row) == target_len:
        return row
    if len(row) < target_len:
        return row + [""] * (target_len - len(row))
    # len(row) > target_len: merge excedentes no último campo
    base = row[: target_len - 1]
    tail = " ".join(x for x in row[target_len - 1 :] if x)
    return base + [tail]


def _is_header_row(row: List[str]) -> bool:
    # Considera header se contém, pelo menos, esses tokens.
    row_n = [_norm(c) for c in row]
    joined = " | ".join(row_n)
    return (
        "nome do campo" in joined
        and re.search(r"\bcampo\b", joined) is not None
        and "tipo" in joined
        and ("descricao" in joined or "descrição" in joined)
        and ("caracteristicas" in joined or "características" in joined)
        and "dbf" in joined
    )


@dataclass
class ExtractStats:
    pages: int = 0
    tables_seen: int = 0  # legacy (modo extract_tables)
    tables_used: int = 0  # legacy (modo extract_tables)
    rows_emitted: int = 0


def _group_words_into_lines(words: list[dict], y_tolerance: float = 3.0) -> list[list[dict]]:
    """
    Agrupa palavras em linhas usando a coordenada vertical (top).
    """
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: list[list[dict]] = []
    current: list[dict] = [words_sorted[0]]
    ref_top = words_sorted[0]["top"]

    for w in words_sorted[1:]:
        if abs(w["top"] - ref_top) <= y_tolerance:
            current.append(w)
            # suaviza referência para lidar com pequenas variações
            ref_top = (ref_top * 0.7) + (w["top"] * 0.3)
        else:
            lines.append(sorted(current, key=lambda x: x["x0"]))
            current = [w]
            ref_top = w["top"]
    lines.append(sorted(current, key=lambda x: x["x0"]))
    return lines


def _find_header_columns_in_line(words_in_line: list[dict]) -> Optional[list[float]]:
    """
    Dada uma linha de palavras, tenta identificar o cabeçalho e retornar os x0
    de início de cada coluna (7 colunas).
    """
    text = " ".join(w["text"] for w in words_in_line)
    nt = _norm(text)
    if not ("nome do campo" in nt and "tipo" in nt and "categoria" in nt and "descricao" in nt and "dbf" in nt):
        return None

    # primeira coluna: "Nome do Campo" -> usa x0 de "Nome"
    x_nome = None
    for w in words_in_line:
        if _norm(w["text"]) == "nome":
            x_nome = w["x0"]
            break
    if x_nome is None:
        return None

    # segunda coluna: "Campo" (há dois "Campo" no header: um no "Nome do Campo" e outro da coluna)
    campos = [w for w in words_in_line if _norm(w["text"]) == "campo"]
    if not campos:
        return None
    # escolhe o "Campo" mais à direita, depois do bloco "Nome do Campo"
    x_campo2 = max(w["x0"] for w in campos if w["x0"] > x_nome + 40) if any(w["x0"] > x_nome + 40 for w in campos) else max(w["x0"] for w in campos)

    def x_of(label: str) -> Optional[float]:
        for w in words_in_line:
            if _norm(w["text"]) == _norm(label):
                return w["x0"]
        return None

    x_tipo = x_of("Tipo")
    x_categoria = x_of("Categoria")
    x_descricao = x_of("Descrição")
    x_caract = x_of("Características")
    x_dbf = x_of("DBF")

    if None in (x_tipo, x_categoria, x_descricao, x_caract, x_dbf):
        return None

    starts = [x_nome, x_campo2, x_tipo, x_categoria, x_descricao, x_caract, x_dbf]
    # garante ordenação crescente
    if any(starts[i] >= starts[i + 1] for i in range(len(starts) - 1)):
        return None
    return starts


def _line_to_columns(words_in_line: list[dict], col_starts: list[float], page_width: float) -> list[str]:
    """
    Converte uma linha de palavras em 7 colunas usando cortes (midpoints) entre os inícios.

    Observação: em vários PDFs, o x0 dos valores pode ficar um pouco à esquerda do x0
    do *rótulo* do cabeçalho; por isso usar o "próximo start" como limite tende a
    jogar valores da última coluna (DBF) para a coluna anterior (Características).
    """
    # cortes padrão: midpoint entre inícios
    cuts = [(col_starts[i] + col_starts[i + 1]) / 2.0 for i in range(6)]
    # ajuste fino do último corte (Características -> DBF):
    # - se usarmos o midpoint puro, parte do DBF pode cair em Características
    # - se trazermos demais para a esquerda, texto de Características cai em DBF
    # então colocamos o corte mais perto do "DBF" (80% do caminho).
    last_left = col_starts[5]
    last_right = col_starts[6]
    cut_dbf = last_left + (last_right - last_left) * 0.8
    # clamp defensivo
    cut_dbf = max(last_left + 20, min(last_right - 10, cut_dbf))
    cuts[5] = cut_dbf
    cols: list[list[str]] = [[] for _ in range(7)]
    for w in words_in_line:
        x0 = w["x0"]
        idx = bisect_right(cuts, x0)
        if 0 <= idx < 7:
            cols[idx].append(w["text"])
    return [_clean_cell(" ".join(c)) for c in cols]


def extract_rows_from_pdf(pdf_path: Path) -> tuple[list[dict], ExtractStats]:
    """
    Extração principal baseada em coordenadas (x0) do cabeçalho e `extract_words()`.
    Isso funciona bem quando as tabelas não têm linhas desenhadas, mas o texto mantém alinhamento.
    """
    rows_out: list[dict] = []
    stats = ExtractStats()

    def is_noise_line(cols: list[str]) -> bool:
        # ignora cabeçalhos/rodapés típicos (podem cair em colunas diferentes dependendo do alinhamento)
        all_text = _norm(" ".join(c for c in cols if c))
        if not all_text:
            return True
        if all_text.startswith("revisado"):
            return True
        if "junho/2015" in all_text:
            return True
        if all_text.startswith("ministerio da saude"):
            return True
        if all_text.startswith("secretaria de vigilancia"):
            return True
        if all_text.startswith("departamento de vigilancia"):
            return True
        if all_text.startswith("centro de informacoes"):
            return True
        if all_text.startswith("gt-sinan"):
            return True
        if all_text.startswith("sistema de informacao"):
            return True
        if all_text.startswith("dicionario de dados"):
            return True
        return False

    def has_valid_dbf(dbf_cell: str) -> bool:
        # DBF costuma ser um ou mais tokens em UPPER_SNAKE_CASE (ex.: DT_NOTIFIC, CO_UNI_EXT)
        tokens = re.findall(r"\b[A-Z][A-Z0-9_]{1,}\b", dbf_cell or "")
        return len(tokens) > 0

    def _line_dbf_candidate(cols: list[str]) -> str:
        """
        Tenta obter o DBF da linha mesmo quando ele "vaza" para outras colunas.
        Retorna o último token plausível em toda a linha.
        """
        stop = {"DBF", "CID", "IBGE", "OMS", "SPM", "MDS", "MEC", "SDH", "CNES"}
        all_text = " ".join(c for c in cols if c)
        tokens = re.findall(r"\b[A-Z][A-Z0-9_]{1,}\b", all_text or "")
        cleaned = [t for t in tokens if t not in stop and len(t) >= 3]
        return cleaned[-1] if cleaned else ""

    def normalize_row(row: list[str]) -> list[str]:
        """
        Normaliza alguns campos que podem vazar texto de colunas adjacentes por causa do PDF:
        - Campo: mantém apenas tokens em snake_case (ex.: co_municipio_residencia)
        - Tipo: tenta extrair um tipo canônico (varchar2(...), varchar(...), number(...), date)
        - DBF: mantém apenas tokens plausíveis (UPPER/underscore), removendo lixo como 'a', 'o', '>'
        """
        nome, campo, tipo, categoria, descricao, caracteristicas, dbf = (row + [""] * 7)[:7]

        # Campo (snake_case)
        campo_tokens = re.findall(r"\b[a-z]{2,}[a-z0-9_]*\b", campo or "")
        # remove tokens comuns que não são nomes de campo
        campo_tokens = [t for t in campo_tokens if t not in {"de", "da", "do", "e", "ou"}]
        campo_norm = "\n".join(dict.fromkeys(campo_tokens))

        # Tipo
        tipo_src = tipo or ""
        tipo_match = re.search(r"\b(varchar2?|number)\s*\([^)]*\)", tipo_src, flags=re.IGNORECASE)
        if tipo_match:
            tipo_norm = tipo_match.group(0).replace(" ", "")
        else:
            tipo_norm = "date" if re.search(r"\bdate\b", tipo_src, flags=re.IGNORECASE) else tipo_src.strip()

        # Descrição: frequentemente começa com "Informar..." ou "Especificar...".
        # A extração pode inverter a ordem de linhas (ex.: "foi através de..." antes de "Informar se...").
        desc_lines = [ln.strip() for ln in (descricao or "").split("\n") if ln.strip()]
        if len(desc_lines) >= 2:
            starters = ("informar", "especificar", "identifica", "data", "nome", "codigo", "código")

            def prio(line: str) -> tuple[int, str]:
                n = _norm(line)
                return (0 if n.startswith(starters) else 1, n)

            # só reordena se houver um candidato "starter" que não esteja na primeira posição
            if prio(desc_lines[0])[0] == 1 and any(prio(ln)[0] == 0 for ln in desc_lines[1:]):
                desc_lines = sorted(desc_lines, key=prio)
        descricao_norm = "\n".join(desc_lines)

        # DBF
        dbf_tokens = re.findall(r"\b[A-Z][A-Z0-9_]{1,}\b", dbf or "")
        stop = {"DBF", "CID", "IBGE", "OMS", "SPM", "MDS", "MEC", "SDH", "CNES"}
        cleaned = []
        for t in dbf_tokens:
            if t in stop:
                continue
            # evita tokens muito curtos (ex.: 'A', 'O') que vêm de vazamento
            if len(t) < 3:
                continue
            cleaned.append(t)
        # Complemento: às vezes o DBF “vaza” para outras colunas por causa do layout.
        # Então, além do que veio na coluna DBF, também coletamos tokens plausíveis
        # em toda a linha e agregamos (sem duplicar).
        all_text = " ".join([nome, campo, tipo, categoria, descricao, caracteristicas, dbf])
        all_tokens = re.findall(r"\b[A-Z][A-Z0-9_]{1,}\b", all_text or "")
        all_clean = [t for t in all_tokens if t not in stop and len(t) >= 3]
        for t in all_clean:
            if t not in cleaned:
                cleaned.append(t)

        dbf_norm = "\n".join(dict.fromkeys(cleaned))

        return [
            (nome or "").strip(),
            campo_norm.strip(),
            (tipo_norm or "").strip(),
            (categoria or "").strip(),
            (descricao_norm or "").strip(),
            (caracteristicas or "").strip(),
            dbf_norm.strip(),
        ]

    with pdfplumber.open(str(pdf_path)) as pdf:
        stats.pages = len(pdf.pages)

        col_starts: Optional[list[float]] = None
        seen_header = False
        current_row: list[str] = [""] * 7

        for page in pdf.pages:
            words = page.extract_words() or []
            lines = _group_words_into_lines(words, y_tolerance=3.0)

            header_top: Optional[float] = None
            if col_starts is None:
                # procura o cabeçalho em alguma linha desta página
                for line in lines:
                    starts = _find_header_columns_in_line(line)
                    if starts is not None:
                        col_starts = starts
                        header_top = min(w["top"] for w in line) if line else None
                        seen_header = True
                        break

            if col_starts is None:
                continue  # ainda não encontrou o header, não dá pra segmentar colunas

            # processa linhas, ignorando o topo antes do header (quando o header está nesta página)
            for line in lines:
                if header_top is not None:
                    line_top = min(w["top"] for w in line) if line else 0
                    if line_top <= header_top + 1:
                        continue

                cols = _line_to_columns(line, col_starts, page.width)

                # pula linha do header, se repetida
                if _is_header_row(cols):
                    continue

                # "Revisado ..." costuma ser separador de seção no PDF, não pertence a nenhum campo.
                # Importante: ele aparece no fim de páginas; se não "fecharmos" o registro atual aqui,
                # o texto do próximo campo pode vazar para o registro anterior.
                line_all_norm = _norm(" ".join(c for c in cols if c))
                if line_all_norm.startswith("revisado") or "junho/2015" in line_all_norm:
                    if current_row[6] and has_valid_dbf(current_row[6]):
                        normed_prev = normalize_row(current_row)
                        if normed_prev[1] and normed_prev[6]:
                            rows_out.append(dict(zip(EXPECTED_COLUMNS, normed_prev)))
                            stats.rows_emitted += 1
                        current_row = [""] * 7
                    continue

                if is_noise_line(cols):
                    continue

                # Início de um novo registro: algumas páginas trazem várias linhas de categoria
                # antes de repetir o DBF. Para não "colar" a categoria do próximo campo no campo anterior,
                # iniciamos um novo registro quando a linha contém:
                # - Nome do campo (ex.: "16. Escolaridade")
                # - Campo (snake_case) e Tipo (varchar2/number/date/...)
                nome0 = (cols[0] or "").strip()
                nome0_norm = _norm(nome0)
                has_nome = bool(re.match(r"^\d+\.", nome0)) or nome0_norm.startswith("(campo interno)")
                has_campo = bool(re.search(r"\b[a-z]{2,}[a-z0-9_]*\b", cols[1] or ""))
                has_tipo = bool(
                    re.search(r"\b(varchar2?|number)\s*\([^)]*\)", cols[2] or "", flags=re.IGNORECASE)
                    or re.search(r"\bdate\b", cols[2] or "", flags=re.IGNORECASE)
                    or re.search(r"\bhora\b", cols[2] or "", flags=re.IGNORECASE)
                )
                if has_nome and has_campo and has_tipo and current_row[6] and has_valid_dbf(current_row[6]):
                    normed_prev = normalize_row(current_row)
                    if normed_prev[1] and normed_prev[6]:
                        rows_out.append(dict(zip(EXPECTED_COLUMNS, normed_prev)))
                        stats.rows_emitted += 1
                    current_row = [""] * 7

                # Regra de delimitação: um registro continua até aparecer um NOVO DBF.
                # Isso permite que linhas de continuação (sem DBF) sejam anexadas ao registro correto
                # (ex.: "Envenenamento, Intoxicação" quebra e "Intoxicação" aparece depois do DBF).
                line_dbf = _line_dbf_candidate(cols)
                if line_dbf and current_row[6] and has_valid_dbf(current_row[6]):
                    # se a linha traz um DBF diferente, fecha o registro anterior antes de acumular esta linha
                    if line_dbf.strip() != current_row[6].strip():
                        normed_prev = normalize_row(current_row)
                        if normed_prev[1] and normed_prev[6]:
                            rows_out.append(dict(zip(EXPECTED_COLUMNS, normed_prev)))
                            stats.rows_emitted += 1
                        current_row = [""] * 7

                # acumula colunas (continuações quebradas entram aqui)
                for i in range(7):
                    if cols[i]:
                        current_row[i] = (current_row[i] + "\n" + cols[i]).strip() if current_row[i] else cols[i]

        # flush final
        if any(current_row) and current_row[6] and has_valid_dbf(current_row[6]):
            normed = normalize_row(current_row)
            if normed[1] and normed[6]:
                rows_out.append(dict(zip(EXPECTED_COLUMNS, normed)))
                stats.rows_emitted += 1

        # sanity: se não achou header, alerta implicitamente via stats
        if not seen_header:
            stats.rows_emitted = 0

    return rows_out, stats


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extrai tabelas do dicionário SINAN (PDF) para JSON.")
    parser.add_argument("--pdf", required=True, help="Caminho do PDF (ex.: docs/dicionario_dados_sinan.pdf)")
    parser.add_argument("--out", required=True, help="Caminho do JSON de saída")
    args = parser.parse_args(list(argv) if argv is not None else None)

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows, stats = extract_rows_from_pdf(pdf_path)

    payload = {
        "fonte_pdf": str(pdf_path),
        "colunas": EXPECTED_COLUMNS,
        "linhas": rows,
        "stats": {
            "pages": stats.pages,
            "tables_seen": stats.tables_seen,
            "tables_used": stats.tables_used,
            "rows_emitted": stats.rows_emitted,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Gerado: {out_path}\n"
        f"   - páginas: {stats.pages}\n"
        f"   - tabelas encontradas: {stats.tables_seen}\n"
        f"   - tabelas usadas (com cabeçalho esperado): {stats.tables_used}\n"
        f"   - linhas emitidas: {stats.rows_emitted}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

