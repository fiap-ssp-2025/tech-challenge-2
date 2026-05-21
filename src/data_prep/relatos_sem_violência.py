# Arquivo gerado e executado pelo gemini(google colab) para criar um dataset de relatos sem violência, 
# com o objetivo de treinar o modelo a reconhecer padrões de linguagem e contextos que não indicam risco, 
# ajudando a reduzir falsos positivos em casos de baixa ou nenhum risco.


import csv
import random

# Base de dados com variações de contextos para expandir os textos naturalmente
sujeitos = ["Meu marido", "Meu namorado", "Meu noivo", "Meu companheiro", "O meu parceiro"]
introducoes = [
    "Estamos juntos há alguns anos e as coisas geralmente vão bem, mas recentemente começamos a esbarrar em um problema que está me desgastando bastante.",
    "Quero desabafar sobre uma situação que tem se repetido na minha rotina e está me deixando bem cansada mentalmente ultimamente.",
    "Amo muito meu parceiro, mas nossa convivência diária tem sido testada por detalhes bobos que acabam virando discussões longas.",
    "Preciso colocar para fora algo que aconteceu essa semana e que me deixou chateada, embora eu saiba que não é o fim do mundo.",
    "Nossa relação é super saudável, mas os pequenos atritos do dia a dia estão começando a acumular e causar um clima chato entre nós."
]

historias_tarefas = [
    "A questão toda é sobre a divisão das tarefas domésticas. Nós dois trabalhamos fora em período integral e chegamos exaustos em casa, mas parece que toda a carga mental de organizar o lar cai apenas sobre as minhas costas. Ontem mesmo combinei com ele que eu faria o jantar e ele ficaria responsável por lavar a louça e organizar a cozinha antes de deitarmos. Eu cozinhei, limpei os balcões enquanto fazia a comida e fui descansar. Para a minha surpresa, quando acordei hoje de manhã para passar o café, a pia estava completamente lotada, cheia de panelas engorduradas e copos espalhados. Quando questionei, ele disse que estava cansado e ia lavar mais tarde. Isso me irrita profundamente porque sinto que preciso ficar mandando ou cobrando o tempo todo para que as coisas básicas aconteçam. Não quero ser a mãe dele, quero apenas um parceiro que divida o teto de forma justa e limpe o que sujou sem precisar de um lembrete.",
    "Discutimos novamente por conta da limpeza do banheiro e da organização geral da casa. Ele sempre se dispõe a ajudar no final de semana, mas a definição dele de 'limpar' é passar um pano úmido rápido e achar que está tudo ótimo. Ele não enxerga a poeira acumulada nos móveis, não limpa os vidros das janelas e finge que não vê o cesto de lixo transbordando na área de serviço. Quando eu peço para ele fazer uma faxina mais detalhada, ele diz que sou perfeccionista demais e que a casa não precisa parecer um hospital. Ficamos horas batendo boca por causa disso no sábado. Ele acabou se isolando no quarto para jogar videogame e eu terminei de limpar a casa sozinha, cheia de raiva. É exaustivo ter que explicar que tapete de banheiro precisa ser lavado e que lençol de cama não se troca uma vez por mês. Falta proatividade da parte dele e isso me desgasta muito.",
    "O grande problema atual é que ele simplesmente não consegue manter nenhum espaço organizado por mais de dez minutos. Ele tira os sapatos sujos assim que entra em casa, mas larga as meias usadas bem no meio do corredor da sala. Quando vai jogar futebol com os amigos, volta com as roupas suadas e joga tudo direto no chão do quarto de hóspedes em vez de colocar no cesto de roupa suja. Ontem ele resolveu inventar uma receita nova na cozinha. O prato ficou ótimo, não posso negar, mas ele deixou o ambiente parecendo que tinha passado um furacão. Tinha farinha no chão, molho respingado nos azulejos e cinco copos diferentes usados espalhados pelas bancadas. Quando falei que ele precisava limpar a bagunça que fez, ele achou ruim e disse que eu reclamo de tudo e não sei valorizar o esforço dele em cozinhar. Acabamos dormindo de costas um para o outro por puro orgulho."
]

historias_financas = [
    "O nosso ponto de discórdia atual é puramente financeiro e envolve o planejamento do nosso futuro. Eu sou uma pessoa extremamente controlada, anoto cada centavo em uma planilha do Excel, acompanho nossos gastos fixos e gosto de separar uma porcentagem do salário para investimentos a longo prazo. Já ele é completamente desorganizado com dinheiro e vive fazendo compras por impulso de coisas que ele vê em anúncios de redes sociais. Esta semana descobri que ele comprou uma televisão gigantesca para o quarto sem me consultar antes, desfalcando uma parte do dinheiro que estávamos poupando para a nossa viagem de férias. Quando fui conversar sobre isso, ele agiu como se não fosse nada demais, dizendo que o dinheiro é dele e que merecia um agrado por trabalhar tanto. Fiquei muito frustrada porque sinto que estou me sacrificando sozinha para construir um patrimônio para nós dois enquanto ele gasta sem pensar no amanhã.",
    "Estamos tendo muitas divergências sobre como equilibrar as contas da casa proporcionalmente, já que nós ganhamos salários bem diferentes. Eu recebo menos e sinto que estou sempre no meu limite financeiro para conseguir pagar a minha parte do aluguel e dos mantimentos. Ele, por ter uma renda maior, quer manter um padrão de vida elevado, sugerindo jantares em restaurantes caros nos finais de semana, viagens caras e planos de trocar de carro ainda este ano. Quando sugiro que fiquemos em casa ou escolhamos opções mais baratas, ele fica chateado e diz que eu estou sendo pão-dura ou estragando os momentos de lazer. Tivemos uma conversa bem tensa ontem à noite sobre isso. Expliquei que não consigo acompanhar o ritmo dele sem contrair dívidas no cartão de crédito. Ele ouviu, mas sinto que ainda não entendeu a gravidade da minha situação financeira e acha que estou apenas reclamando de barriga cheia.",
    "O clima pesou aqui em casa porque descobri que ele emprestou uma quantia considerável de dinheiro para o irmão dele sem falar comigo antes. Nós temos uma conta conjunta para pagar as despesas básicas do apartamento e era exatamente desse fundo que o dinheiro saiu. Quando a conta de luz e a fatura da internet venceram, percebi que o saldo estava abaixo do esperado. Ele justificou dizendo que o irmão estava passando por uma emergência familiar e que devolveria o valor no próximo mês, mas o que me magoou foi a falta de comunicação e transparência. Se dividimos uma vida e uma conta bancária, decisões que afetam o nosso orçamento doméstico deveriam ser tomadas em conjunto. Ele achou que eu estava sendo insensível com o problema da família dele, e a discussão acabou desviando do foco principal, que era o respeito aos nossos combinados financeiros."
]

historias_ciumes = [
    "Tivemos uma discussão boba ontem por causa de redes sociais e pura insegurança da parte dele. Eu postei uma foto antiga com várias amigas da faculdade e um antigo colega de classe acabou deixando um comentário elogioso simples, dizendo que eu estava radiante. Meu namorado viu o comentário e mudou completamente o comportamento comigo, ficando de bico o resto da noite. Quando perguntei o que estava acontecendo, ele começou a questionar quem era esse rapaz, se nós já tínhamos tido algo no passado e por que ele se sentia no direito de comentar nas minhas fotos. Achei a atitude dele extremamente infantil. Expliquei calmamente que era apenas um amigo de anos atrás e que o comentário não tinha segundas intenções. Ele fechou a cara, disse que homem conhece homem e que sabe quais são as intenções dos outros. Passamos o final de semana sem nos falar direito por causa de uma bobagem dessas, o que me deixa muito cansada.",
    "O problema é o ciúme bobo que ele sente do meu ambiente de trabalho. Sou a única mulher em uma equipe de engenharia e é perfeitamente normal que eu interaja com meus colegas. Esta semana, mudamos a foto de perfil do grupo corporativo no WhatsApp e um dos rapazes comentou no meu LinkedIn parabenizando por um projeto recente. Meu noivo viu a notificação no meu celular e ficou super incomodado. Ele começou a insinuar que o colega estava sendo simpático demais e perguntou se precisávamos mesmo conversar fora do horário de expediente. Expliquei que era uma rede profissional e que não havia nada de errado ali, mas ele se trancou no escritório dele e passou a noite emburrado. Fico chateada porque confio plenamente nele e sinto que essas bobeiras mostram uma falta de confiança em mim, mesmo eu nunca tendo dado nenhum motivo para isso acontecer.",
    "Ontem fomos a um jantar de aniversário de uma amiga minha e o garçom que nos atendeu foi extremamente educado e simpático comigo. Assim que o rapaz se afastou da mesa, meu parceiro mudou o tom de voz e começou a fazer piadas sem graça, insinuando que o funcionário estava dando em cima de mim. O clima do jantar azedou completamente a partir dali. Ele passou o resto da noite sério, respondendo apenas com monossílabas e olhando feio para o rapaz toda vez que ele trazia algo para a mesa. Quando chegamos em casa, reclamei da postura dele e disse que foi uma vergonha passar por aquilo. Ele se defendeu dizendo que eu fui simpática demais com o garçom e que deveria ter cortado o assunto. É exaustivo ter que lidar com essas crises de insegurança besta em momentos que deveriam ser de pura diversão e relaxamento."
]

conclusoes = [
    "Espero que consigamos sentar e conversar de forma mais madura nos próximos dias para resolver isso de vez.",
    "Sei que todo casal passa por fases de ajuste, mas precisaremos alinhar nossas expectativas para o relacionamento funcionar.",
    "No fundo sei que é apenas uma fase de estresse acumulado, mas precisava desabafar para não guardar esse sentimento ruim.",
    "Vamos tentar conversar hoje à noite com a cabeça mais fria para restabelecer nossos combinados cotidianos.",
    "Amo nossa vida juntos, mas esses pequenos ruídos de comunicação precisam ser resolvidos para o bem do nosso futuro."
]

# Gerar os 1000 relatos mesclando as estruturas para garantir tamanho (~160 palavras) e aleatoriedade
dados_csv = []
todas_historias = historias_tarefas + historias_financas + historias_ciumes

for i in range(1000):
    sub = random.choice(sujeitos)
    intro = random.choice(introducoes)
    historia = random.choice(todas_historias)
    conc = random.choice(conclusoes)
    
    # Monta o relato completo
    relato_completo = f"{intro} {historia} {sub} e eu precisamos melhorar nisso. {conc}"
    
    # Adiciona na estrutura pedida: [Texto, Relato, Alvo/Target]
    dados_csv.append(["SEM_VIOLÊNCIA", relato_completo, 0])

# Gravar o arquivo CSV
nome_arquivo = 'relatos_treinamento.csv'
with open(nome_arquivo, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # Escreve o cabeçalho (opcional, remova se seu modelo não usar)
    writer.writerow(["Classe", "Relato", "Target"])
    writer.writerows(dados_csv)

print(f"Sucesso! Arquivo '{nome_arquivo}' gerado com {len(dados_csv)} relatos.")
