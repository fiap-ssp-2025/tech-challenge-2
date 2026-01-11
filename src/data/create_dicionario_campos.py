"""
Script para criar o dicionário completo de campos do SINAN - Violência Interpessoal/Autoprovocada.

Este script gera um dicionário com todas as 160 colunas do dataset, incluindo:
- nome: Nome completo do campo
- descricao: Descrição do campo
- tipo: Tipo de dado no SINAN
- obrigatorio: Se o campo é obrigatório
- categoria: Categorias possíveis (quando aplicável)
- relevante: Campo booleano para marcar campos relevantes para análise (inicialmente False)

Fonte: Dicionário de Dados SINAN NET - Versão 5.0/Patch 5.1
"""

# Lista completa de todas as colunas do dataset SINAN VIOL
COLUNAS_SINAN = [
    'AG_AMEACA', 'AG_CORTE', 'AG_ENFOR', 'AG_ENVEN', 'AG_ESPEC', 'AG_FOGO', 
    'AG_FORCA', 'AG_OBJETO', 'AG_OUTROS', 'AG_QUENTE', 'ANO_NASC', 'ASSIST_SOC', 
    'ATEND_MULH', 'AUTOR_ALCO', 'AUTOR_SEXO', 'CICL_VID', 'CIRC_LESAO', 
    'CLASSI_FIN', 'CONS_ABORT', 'CONS_COMP', 'CONS_DST', 'CONS_ESPEC', 
    'CONS_ESTRE', 'CONS_GRAV', 'CONS_IDO', 'CONS_MENT', 'CONS_OUTR', 'CONS_SUIC', 
    'CONS_TUTEL', 'CS_ESCOL_N', 'CS_GESTANT', 'CS_RACA', 'CS_SEXO', 'DEFEN_PUBL', 
    'DEF_AUDITI', 'DEF_ESPEC', 'DEF_FISICA', 'DEF_MENTAL', 'DEF_OUT', 'DEF_TRANS', 
    'DEF_VISUAL', 'DELEG', 'DELEG_CRIA', 'DELEG_IDOS', 'DELEG_MULH', 'DIR_HUMAN', 
    'DT_DIGITA', 'DT_ENCERRA', 'DT_INVEST', 'DT_NOTIFIC', 'DT_OBITO', 'DT_OCOR', 
    'DT_TRANSDM', 'DT_TRANSRM', 'DT_TRANSRS', 'DT_TRANSSE', 'DT_TRANSSM', 
    'DT_TRANSUS', 'ENC_ABRIGO', 'ENC_CREAS', 'ENC_DEAM', 'ENC_DELEG', 'ENC_DPCA', 
    'ENC_ESPEC', 'ENC_IML', 'ENC_MPU', 'ENC_MULHER', 'ENC_OUTR', 'ENC_SAUDE', 
    'ENC_SENTIN', 'ENC_TUTELA', 'ENC_VARA', 'EVOLUCAO', 'HORA_OCOR', 'IDENT_GEN', 
    'ID_AGRAVO', 'ID_MN_OCOR', 'ID_MN_RESI', 'ID_MUNICIP', 'ID_OCUPA_N', 'ID_PAIS', 
    'ID_UNIDADE', 'INFAN_JUV', 'LESAO_CORP', 'LESAO_ESPE', 'LESAO_NAT', 
    'LES_AUTOP', 'LOCAL_ESPE', 'LOCAL_OCOR', 'MPU', 'NDUPLIC', 'NUM_ENVOLV', 
    'NU_ANO', 'NU_IDADE_N', 'ORIENT_SEX', 'OUT_VEZES', 'PEN_ANAL', 'PEN_ORAL', 
    'PEN_VAGINA', 'PROC_ABORT', 'PROC_CONTR', 'PROC_DST', 'PROC_HEPB', 'PROC_HIV', 
    'PROC_SANG', 'PROC_SEMEN', 'PROC_VAGIN', 'REDE_EDUCA', 'REDE_SAU', 'REL_CAT', 
    'REL_CONHEC', 'REL_CONJ', 'REL_CUIDA', 'REL_DESCO', 'REL_ESPEC', 'REL_EXCON', 
    'REL_EXNAM', 'REL_FILHO', 'REL_INST', 'REL_IRMAO', 'REL_MAD', 'REL_MAE', 
    'REL_NAMO', 'REL_OUTROS', 'REL_PAD', 'REL_PAI', 'REL_PATRAO', 'REL_POL', 
    'REL_PROPRI', 'REL_SEXUAL', 'REL_TRAB', 'SEM_NOT', 'SEM_PRI', 'SEX_ASSEDI', 
    'SEX_ESPEC', 'SEX_ESTUPR', 'SEX_EXPLO', 'SEX_OUTRO', 'SEX_PORNO', 'SEX_PUDOR', 
    'SG_UF', 'SG_UF_NOT', 'SG_UF_OCOR', 'SIT_CONJUG', 'TPUNINOT', 'TP_NOT', 
    'TRAN_COMP', 'TRAN_MENT', 'VIOL_ESPEC', 'VIOL_FINAN', 'VIOL_FISIC', 
    'VIOL_INFAN', 'VIOL_LEGAL', 'VIOL_MOTIV', 'VIOL_NEGLI', 'VIOL_OUTR', 
    'VIOL_PSICO', 'VIOL_SEXU', 'VIOL_TORT', 'VIOL_TRAF'
]


def criar_dicionario_campos():
    """
    Cria o dicionário completo de campos do SINAN.
    
    Returns:
    --------
    dict: Dicionário com todas as colunas e suas informações
    """
    dicionario = {}
    
    # Campos conhecidos do dicionário oficial (com informações completas)
    campos_conhecidos = {
        'NU_NOTIFIC': {
            'nome': 'N° da Notificação',
            'descricao': 'Número da Notificação - Campo Chave para identificação do registro no sistema',
            'tipo': 'varchar2(7)',
            'obrigatorio': True,
            'categoria': None
        },
        'TP_NOT': {
            'nome': 'Tipo de Notificação',
            'descricao': 'Identifica o tipo da notificação',
            'tipo': 'varchar2(1)',
            'obrigatorio': True,
            'categoria': {
                '1': 'Negativa',
                '2': 'Individual',
                '3': 'Surto',
                '4': 'Agregado'
            }
        },
        'ID_AGRAVO': {
            'nome': 'Agravo',
            'descricao': 'Nome e código do agravo notificado segundo CID-10',
            'tipo': 'varchar2(4)',
            'obrigatorio': True,
            'categoria': 'Tabela de agravos do sistema com códigos CID-10'
        },
        'DT_NOTIFIC': {
            'nome': 'Data da Notificação',
            'descricao': 'Data de preenchimento da ficha de notificação (dd/mm/aaaa)',
            'tipo': 'date',
            'obrigatorio': True,
            'categoria': 'dd/mm/aaaa'
        },
        'SEM_NOT': {
            'nome': 'Semana epidemiológica da notificação',
            'descricao': 'Semanas do calendário epidemiológico padronizado (AAAASS)',
            'tipo': 'varchar2(6)',
            'obrigatorio': False,
            'categoria': 'Preenchida automaticamente'
        },
        'NU_ANO': {
            'nome': 'Ano da notificação',
            'descricao': 'Ano da notificação - Variável interna preenchida pelo sistema',
            'tipo': 'varchar(4)',
            'obrigatorio': False,
            'categoria': None
        },
        'SG_UF_NOT': {
            'nome': 'UF de Notificação',
            'descricao': 'Sigla da Unidade Federativa onde está localizada a unidade de saúde',
            'tipo': 'varchar2(2)',
            'obrigatorio': True,
            'categoria': 'Tabela com Códigos e siglas padronizados pelo IBGE'
        },
        'ID_MUNICIP': {
            'nome': 'Município de Notificação',
            'descricao': 'Código do município onde está localizada a unidade de saúde',
            'tipo': 'varchar2(6)',
            'obrigatorio': True,
            'categoria': 'Tabela com Código e nome dos municípios do cadastro do IBGE'
        },
        'DT_OCOR': {
            'nome': 'Data da ocorrência da violência',
            'descricao': 'Data da ocorrência da violência (dd/mm/aaaa)',
            'tipo': 'date',
            'obrigatorio': True,
            'categoria': 'Data menor ou igual (<=) a Data de Notificação'
        },
        'AUTOR_SEXO': {
            'nome': 'Sexo do provável autor da violência',
            'descricao': 'Informar o sexo do provável autor da agressão',
            'tipo': 'varchar2(1)',
            'obrigatorio': True,
            'categoria': {
                '1': 'Masculino',
                '2': 'Feminino',
                '3': 'Ambos os sexos',
                '9': 'Ignorado'
            }
        },
        'ENC_SAUDE': {
            'nome': 'Encaminhamento - Rede da Saúde',
            'descricao': 'Informar se houve encaminhamento no setor da rede da saúde',
            'tipo': 'varchar2(1)',
            'obrigatorio': True,
            'categoria': {
                '1': 'Sim',
                '2': 'Não',
                '9': 'Ignorado'
            }
        },
        'REL_TRAB': {
            'nome': 'Violência relacionada ao trabalho',
            'descricao': 'Informar se ocorreu violência relacionada ao trabalho',
            'tipo': 'varchar2(1)',
            'obrigatorio': False,
            'categoria': {
                '1': 'Sim',
                '2': 'Não',
                '9': 'Ignorado'
            }
        },
        'CIRC_LESAO': {
            'nome': 'Circunstância da lesão',
            'descricao': 'Nome e código do agravo notificado segundo CID-10 - CAPITULO XX (VO1 a Y98)',
            'tipo': 'varchar2(5)',
            'obrigatorio': False,
            'categoria': 'Tabela de agravos do sistema com códigos CID-10'
        },
        'DT_ENCERRA': {
            'nome': 'Data de encerramento',
            'descricao': 'Data de encerramento do caso (dd/mm/aaaa)',
            'tipo': 'date',
            'obrigatorio': False,
            'categoria': 'Data >= data da notificação'
        }
    }
    
    # Cria dicionário para todas as colunas
    for coluna in COLUNAS_SINAN:
        if coluna in campos_conhecidos:
            # Usa informações conhecidas
            dicionario[coluna] = campos_conhecidos[coluna].copy()
        else:
            # Cria estrutura básica para campos não mapeados
            dicionario[coluna] = {
                'nome': coluna,  # Nome padrão (pode ser atualizado depois)
                'descricao': f'Campo {coluna} - Descrição a ser preenchida conforme dicionário SINAN',
                'tipo': 'varchar2',  # Tipo padrão
                'obrigatorio': False,  # Padrão: não obrigatório
                'categoria': None
            }
        
        # Adiciona campo relevante (inicialmente False)
        dicionario[coluna]['relevante'] = False
    
    return dicionario


def salvar_dicionario(dicionario, caminho_arquivo='dicionario_campos_sinan.py'):
    """
    Salva o dicionário em um arquivo Python.
    
    Parameters:
    -----------
    dicionario : dict
        Dicionário de campos
    caminho_arquivo : str
        Caminho do arquivo para salvar
    """
    import json
    
    # Salva como JSON também (mais fácil de ler)
    caminho_json = caminho_arquivo.replace('.py', '.json')
    
    with open(caminho_json, 'w', encoding='utf-8') as f:
        json.dump(dicionario, f, ensure_ascii=False, indent=2)
    
    # Salva como Python (dicionário)
    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        f.write('"""\n')
        f.write('Dicionário completo de campos do SINAN - Violência Interpessoal/Autoprovocada\n')
        f.write('Fonte: Dicionário de Dados SINAN NET - Versão 5.0/Patch 5.1\n')
        f.write('Gerado automaticamente\n')
        f.write('"""\n\n')
        f.write('DICIONARIO_CAMPOS = ')
        
        # Converte para string formatada
        import pprint
        f.write(pprint.pformat(dicionario, width=120, indent=4))
    
    print(f"✅ Dicionário salvo em: {caminho_arquivo}")
    print(f"✅ Dicionário JSON salvo em: {caminho_json}")
    print(f"📊 Total de campos: {len(dicionario)}")


if __name__ == '__main__':
    print("🔨 Criando dicionário completo de campos SINAN...")
    print("="*80)
    
    dicionario = criar_dicionario_campos()
    
    print(f"\n✅ Dicionário criado com {len(dicionario)} campos")
    campos_completos = sum(1 for v in dicionario.values() if 'Campo' not in v.get('descricao', ''))
    print(f"📋 Campos com informações completas: {campos_completos}")
    campos_nao_relevantes = sum(1 for v in dicionario.values() if not v.get('relevante', True))
    print(f"📋 Campos com relevante=False: {campos_nao_relevantes}")
    
    # Salva o dicionário
    from pathlib import Path
    base_path = Path(__file__).parent.parent.parent
    caminho_salvar = base_path / 'src' / 'data' / 'dicionario_campos_sinan.py'
    caminho_salvar.parent.mkdir(parents=True, exist_ok=True)
    
    salvar_dicionario(dicionario, str(caminho_salvar))
    
    print("\n" + "="*80)
    print("📝 Próximos passos:")
    print("   1. Revise o dicionário e preencha descrições faltantes")
    print("   2. Marque campos relevantes como relevante=True")
    print("   3. Atualize tipos e categorias conforme necessário")
    print("="*80)
