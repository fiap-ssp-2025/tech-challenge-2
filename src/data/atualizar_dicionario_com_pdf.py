"""
Script para atualizar o dicionário de campos com informações do PDF do SINAN.

Este script cruza os dados extraídos do dicionário oficial (PDF) com o JSON atual e
atualiza o campo `descricao` usando o conteúdo da coluna "Características" do PDF.

Fonte do PDF (já convertida em JSON):
  - src/data/dicionario_dados_sinan_tabelas.json
"""

import json
from pathlib import Path

def atualizar_dicionario_com_caracteristicas():
    """
    Atualiza o dicionário JSON com as características do PDF.
    """
    # Caminho do arquivo JSON
    base_path = Path(__file__).parent.parent.parent
    json_path = base_path / 'src' / 'data' / 'dicionario_campos_sinan.json'
    py_path = base_path / 'src' / 'data' / 'dicionario_campos_sinan.py'
    pdf_json_path = base_path / 'src' / 'data' / 'dicionario_dados_sinan_tabelas.json'
    
    # Carrega o JSON atual
    print(f"📂 Carregando JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        dicionario = json.load(f)

    # Carrega as tabelas extraídas do PDF (já em JSON)
    print(f"📂 Carregando tabelas do PDF (JSON): {pdf_json_path}")
    with open(pdf_json_path, 'r', encoding='utf-8') as f:
        pdf_data = json.load(f)
    linhas = pdf_data.get('linhas', [])
    
    # Contadores
    atualizados = 0
    nao_encontrados = set()
    pulados_sem_caracteristicas = 0
    
    # Atualiza as descrições com as características do PDF
    print("\n🔄 Atualizando descrições com características do PDF...")
    print("="*80)

    for item in linhas:
        # Requisito: copiar "Descrição" (da tabela do PDF) para o campo `descricao` do dicionário
        descricao_pdf = (item.get('Descrição') or '').strip()
        if not descricao_pdf:
            pulados_sem_caracteristicas += 1
            continue

        # DBF pode vir com múltiplos códigos separados por \n
        dbf_raw = (item.get('DBF') or '').strip()
        dbf_codes = [c.strip() for c in dbf_raw.split('\n') if c.strip()]
        if not dbf_codes:
            continue

        for code in dbf_codes:
            if code in dicionario:
                dicionario[code]['descricao'] = descricao_pdf
                atualizados += 1
            else:
                nao_encontrados.add(code)
    
    # Salva o JSON atualizado
    print(f"\n💾 Salvando JSON atualizado...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dicionario, f, ensure_ascii=False, indent=2)
    
    # Atualiza também o arquivo Python
    print(f"💾 Atualizando arquivo Python...")
    atualizar_arquivo_python(dicionario, py_path)
    
    # Resumo
    print("\n" + "="*80)
    print("📊 RESUMO DA ATUALIZAÇÃO")
    print("="*80)
    print(f"✅ Campos atualizados: {atualizados}")
    print(f"⏭️  Linhas do PDF puladas (sem Características): {pulados_sem_caracteristicas}")
    if nao_encontrados:
        nao_encontrados_list = sorted(nao_encontrados)
        print(f"⚠️  Códigos DBF não encontrados no JSON: {len(nao_encontrados_list)}")
        for campo in nao_encontrados_list[:10]:  # Mostra apenas os primeiros 10
            print(f"   - {campo}")
        if len(nao_encontrados_list) > 10:
            print(f"   ... e mais {len(nao_encontrados_list) - 10} campos")
    print(f"📁 Arquivo JSON: {json_path}")
    print(f"📁 Arquivo Python: {py_path}")
    print("="*80)


def atualizar_arquivo_python(dicionario, py_path):
    """
    Atualiza o arquivo Python com o dicionário atualizado.
    """
    import pprint
    
    with open(py_path, 'w', encoding='utf-8') as f:
        f.write('"""\n')
        f.write('Dicionário completo de campos do SINAN - Violência Interpessoal/Autoprovocada\n')
        f.write('Fonte: Dicionário de Dados SINAN NET - Versão 5.0/Patch 5.1\n')
        f.write('Gerado automaticamente - Atualizado com características do PDF\n')
        f.write('"""\n\n')
        f.write('DICIONARIO_CAMPOS = ')
        
        # Converte para string formatada
        f.write(pprint.pformat(dicionario, width=120, indent=4))


if __name__ == '__main__':
    print("🔨 Atualizando dicionário com características do PDF...")
    print("="*80)
    atualizar_dicionario_com_caracteristicas()
    print("\n✅ Atualização concluída!")
