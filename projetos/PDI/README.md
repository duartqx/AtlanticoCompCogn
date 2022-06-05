# Atlântico Bootcamp
## PDI
### VISÃO GERAL
Uma empresa contratante deseja estabelecer um sistema automático para medir atributos de folhas de plantas. Neste contexto deve ser desenvolvida uma função que recebe o caminho de um diretório com qualquer quantidade de imagens de folhas e retornar esses atributos específicos. No entanto, para validar a funcionalidade desta função é necessário calcular fazer uma análise quantitativa do desempenho do algoritmo que será responsável por essa tarefa.

### OBJETIVOS

1. Definir padrão de aquisição das imagens de folhas.
2. Realizar aquisição de 100 exemplos no padrão estabelecido pela equipe de desenvolvimento.
3. Realizar um padrão ouro de segmentação de 20 exemplos.
4. Determinar padrão ouro da maior e menor largura dos exemplos definidos no padrão ouro de segmentação.
5. Determinar padrão ouro do valor da área dos exemplos do padrão ouro de segmentação.
6. Armazenar as informações dos itens 3,4,5 em um .csv.
7. Implementar 3 técnicas de segmentação e aplicar nos 20 exemplos com padrão ouro definido.
8. Calcular métricas para avaliação da qualidade da segmentação: IOU (𝑖𝑛𝑡𝑒𝑟𝑠𝑒𝑐𝑡𝑖𝑜𝑛 𝑜𝑣𝑒𝑟 𝑢𝑛𝑖𝑜𝑛)
	> dica: https://towardsdatascience.com/intersection-over-union-iou-calculation-for-evaluating-an-image-segmentation-model-8b22e2e84686
9. Com  a melhor técnica realizar  segmentação de todas as imagens e armazenar o resultado (imagem original ao lado da segmentação) em drive.
10. Utilizando a técnica de segmentação escolhida, determinar os atributos de larguras e áreas, em unidade de pixel. Salvar resultado em .csv
11. Gerar um relatório explicando a metodologia implementada e resultados quantitativos obtidos, além de principais imagens exemplos.
12. Gerar apresentação em slides do relatório, para dia 11 de junho ser apresentado à turma e discutirmos as diferentes soluções e resultados de cada squad.
