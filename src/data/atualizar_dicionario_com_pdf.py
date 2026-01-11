"""
Script para atualizar o dicionário de campos com informações do PDF do SINAN.

Este script cruza os dados do dicionário oficial (PDF) com o JSON atual e atualiza
as descrições com as informações da coluna "Características" do PDF.
"""

import json
from pathlib import Path

# Mapeamento de campos do PDF com suas características (coluna "Características")
# Baseado no Dicionário de Dados SINAN NET - Versão 5.0/Patch 5.1
# Nota: Alguns campos do PDF têm nomes diferentes no JSON (mapeamento abaixo)
CARACTERISTICAS_PDF = {
    # Campos principais
    'TP_NOT': 'Campo Obrigatório',
    'ID_AGRAVO': 'Campo Chave. Preenchendo o código, a descrição é preenchida automaticamente, e vice-versa. Ao exportar, é retirado o ponto',
    'DT_NOTIFIC': 'Campo Chave',
    'SEM_NOT': 'Preenchida automaticamente, a partir da data de notificação (AAAASS)',
    'NU_ANO': 'Variável interna preenchida pelo sistema a partir da data de notificação',
    'SG_UF_NOT': 'Campo Obrigatório',
    'ID_MUNICIP': 'Campo Chave. Preenchendo o código, a descrição é preenchida automaticamente, e vice-versa',
    'ID_UNIDADE': 'Campo Obrigatório. Ao preencher o código, a descrição é preenchida automaticamente e vice-versa',
    'DT_OCOR': 'Campo Obrigatório. Data menor ou igual (<=) a Data de Notificação',
    'SEM_PRI': 'Preenchida automaticamente, a partir da data de primeiros sintomas data do diagnóstico. (AAAASS)',
    
    # Autor da violência
    'AUTOR_SEXO': 'Campo obrigatório',
    'AUTOR_ALCO': '',  # Sem características específicas no PDF visível
    'CICL_VID': 'Campo obrigatório',  # No JSON é CICL_VID, no PDF é CICL_VID_AUTOR
    
    # Encaminhamentos
    'ENC_SAUDE': 'Campo obrigatório',
    'ASSIST_SOC': 'Campo obrigatório',
    'REDE_EDUCA': 'Campo obrigatório',
    'ATEND_MULH': 'Campo obrigatório',
    'CONS_TUTEL': 'Campo obrigatório',
    'CONS_IDO': 'Campo obrigatório',
    'DELEG_IDOS': 'Campo obrigatório',  # No JSON é DELEG_IDOS, no PDF é DELEG_IDOSO
    'DIR_HUMAN': 'Campo obrigatório',
    'MPU': 'Campo obrigatório',
    'DELEG_CRIA': 'Campo obrigatório',
    'DELEG_MULH': 'Campo obrigatório',
    'DELEG': 'Campo obrigatório',
    'INFAN_JUV': 'Campo obrigatório',
    'DEFEN_PUBL': 'Campo obrigatório',
    
    # Violência relacionada ao trabalho
    'REL_TRAB': 'Campo Essencial. Se categoria=2 ou 9 pular para o campo 68. Circunstância da lesão',
    'REL_CAT': 'Categoria=8 se campo 66. Violência relacionada ao trabalho for = 2 ou 9. Se campo 66. Violência relacionada ao trabalho for = 1 não permitir a categoria 8. Não se aplica',
    
    # Outros campos importantes
    'CIRC_LESAO': 'Campo Essencial',
    'DT_ENCERRA': 'Campo >= data da notificação',
    'REL_OUTROS': 'Campo obrigatório. Se categoria=2 ou 9, pular para campo 62. Sexo do provável autor da agressão',
    'REL_ESPEC': 'Campo Obrigatório se campo 61. Relação com a pessoa atendida – Outros =1'
}


def atualizar_dicionario_com_caracteristicas():
    """
    Atualiza o dicionário JSON com as características do PDF.
    """
    # Caminho do arquivo JSON
    base_path = Path(__file__).parent.parent.parent
    json_path = base_path / 'src' / 'data' / 'dicionario_campos_sinan.json'
    py_path = base_path / 'src' / 'data' / 'dicionario_campos_sinan.py'
    
    # Carrega o JSON atual
    print(f"📂 Carregando JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        dicionario = json.load(f)
    
    # Contadores
    atualizados = 0
    nao_encontrados = []
    
    # Atualiza as descrições com as características do PDF
    print("\n🔄 Atualizando descrições com características do PDF...")
    print("="*80)
    
    for campo, caracteristicas in CARACTERISTICAS_PDF.items():
        if campo in dicionario:
            # Se já tem uma descrição completa, adiciona as características
            descricao_atual = dicionario[campo].get('descricao', '')
            
            # Se a descrição atual é apenas o placeholder, substitui completamente
            if 'Descrição a ser preenchida conforme dicionário SINAN' in descricao_atual:
                if caracteristicas:
                    # Se tem características, usa a descrição existente (se houver) + características
                    nome_campo = dicionario[campo].get('nome', campo)
                    dicionario[campo]['descricao'] = f"{nome_campo}. {caracteristicas}"
                else:
                    # Mantém o placeholder se não houver características
                    pass
            else:
                # Se já tem descrição, adiciona as características no final
                if caracteristicas:
                    if caracteristicas not in descricao_atual:
                        dicionario[campo]['descricao'] = f"{descricao_atual}. {caracteristicas}"
            
            # Atualiza obrigatorio baseado nas características
            if 'Campo Obrigatório' in caracteristicas or 'Campo obrigatório' in caracteristicas:
                dicionario[campo]['obrigatorio'] = True
            elif 'Campo Chave' in caracteristicas:
                dicionario[campo]['obrigatorio'] = True
            elif 'Campo Essencial' in caracteristicas:
                dicionario[campo]['obrigatorio'] = False  # Essencial não é obrigatório
            
            atualizados += 1
            print(f"✅ {campo}: Atualizado")
        else:
            nao_encontrados.append(campo)
    
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
    if nao_encontrados:
        print(f"⚠️  Campos não encontrados no JSON: {len(nao_encontrados)}")
        for campo in nao_encontrados[:10]:  # Mostra apenas os primeiros 10
            print(f"   - {campo}")
        if len(nao_encontrados) > 10:
            print(f"   ... e mais {len(nao_encontrados) - 10} campos")
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
