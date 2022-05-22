import cv2
# Leitura da imagem com imread()
image = cv2.imread('entrada.jpg')
print('Largura em pixels:', end='')
print(image.shape[1])
# Largura da imagem
print('Altura em pixels:', end='')
print(image.shape[0])
# Altura da imagem
print('Qtde de canais:', end='')
print(image.shape[2])
# Mostra a imagem com a função imshow
cv2.imshow('Nome da janela', image)
cv2.waitKey(0) # Espera pressionar qualquer tecla
# Salvar a imagem no disco com função imwrite()
cv2.imwrite('saida.jpg', image)
