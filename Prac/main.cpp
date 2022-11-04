#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <Math.h>

using namespace cv;
using namespace std;

//Filtro Gaussiano
Mat implementarGauss(Mat imagen, int filas, int columnas);
double** generarMascaraGauss(int N, double sigma);
double calcularGauss(int x, int y, double sigma);
double getSumaKernel(double** kernel, int N);
Mat imagenRedim(Mat imagen, int filas, int columnas, int N);

//Ecualizacion
int getFrequencyOf(int value, Mat imagen);
int* getImageCDFArray(Mat imagen);
Mat ecualizarImagen(Mat imagen);
int getMinCDF(int* cdfArray);
int getMaxCDF(int* cdfArray);

//Filtro Sobel
double** getMascXSobel(void);
double** getMascYSobel(void);
Mat setMascaraSobel(Mat imagen, double** mascara);
Mat sobelFunction(Mat X, Mat Y, int umbral);
Mat filtroSobel(Mat imagen);
Mat redimensionarGray(Mat imagen, int filas, int columnas, int N);
int convolucionSobel(int fila, int columna, double** mascara, Mat imagen);

//CANNY
Mat non_MS(Mat magnitudes, double** angulos);
int getDirByArg(double angulo);
double** angulos(Mat imgX, Mat imgY);
Mat primerUmbralado(Mat imagenNSM, int lowValue, int highValue);
int isBorder(Mat imagen, int x, int y);

//Operadores
void mostrarImagen(char nombreVentana[], Mat imagen);
void imprimirMatriz(double** mascara, int N);

//Funciones de Utilidad
Mat aplicarFiltro(Mat imagen, double** mascara, int imgX, int imgY, int N, int dimImgX, int dimImgY, double suma);

int main() {
	/*****************DECLARACION DE LAS VARIABLES GENERALES*****************/
	char NombreImagen[] = "lena.jpg"; //Indicamos la ubicacion de la imagen en nuestro equipo
	Mat imagen; //Declaramos la variable que almacenara la imagen abierta
	
	/************************ABRIMOS LA IMAGEN*******************************/
	imagen = imread(NombreImagen);
	if (!imagen.data) {
		cout << "ERROR AL CARGAR LA IMAGEN: " << NombreImagen << endl;
		exit(1);
	}
	char nombreVentanaOriginal[] = "Imagen Original";
	mostrarImagen(nombreVentanaOriginal, imagen);
	/***********************************************************************/

	/********PROCESOS*********/
	//Dimensiones de la imagen original
	int filaOriginal = imagen.rows;
	int columna_original = imagen.cols;

	/********************IMPLEMENTAMOS GAUSS************************/
	Mat imagenGauss = implementarGauss(imagen, filaOriginal, columna_original);

	/********************IMPLEMENTAMOS ECUALIZACION*****************/
	Mat imagenEcualizada = ecualizarImagen(imagenGauss);

	//Aplicamos Sobel
	Mat imagenSobel = filtroSobel(imagenEcualizada);

	waitKey(0);
	return 1;
}

//Ecualizacion Funciones
Mat ecualizarImagen(Mat imagen) {
	int filas = imagen.rows;
	int columnas = imagen.cols;
	int pixelVal, newPixelVal;

	int* cdf = getImageCDFArray(imagen);
	
	Mat nuevaImagen(filas, columnas, CV_8UC1);

	for (int i = 0; i < filas; i++) {
		for (int j = 0; j < columnas; j++) {
			pixelVal = imagen.at<uchar>(Point(i, j));
			newPixelVal = cdf[pixelVal];
			nuevaImagen.at<uchar>(Point(i, j)) = uchar(newPixelVal);
		}
	}

	char nombreImgEcualizada[] = "Imagen_Ecualizada.xls";
	mostrarImagen(nombreImgEcualizada, nuevaImagen);

	return nuevaImagen;
}
int* getImageCDFArray(Mat imagen) {
	int* frecuencias = new int[256]; //Arreglo Histograma qe almacenara la frecuencia de cada una de las intensidades
	int* cdfArray = new int[256]; //Primer CDF
	int* cdfArray_N = new int[256]; //CDF Normalizado

	//Obtenemos todas las frecuencias
	cout << "HISTOGRAMA DE FRECUENCIAS" << endl;
	for (int i = 0; i < 256; i++) {
		frecuencias[i] = getFrequencyOf(i, imagen);
		cout << "[" << frecuencias[i] << "]";
	}

	//Obtenemos el arrgelo de acumulative
	cdfArray[0] = frecuencias[0];
	for (int i = 1; i < 256; i++) {
		cdfArray[i] = cdfArray[i - 1] + frecuencias[i];
	}

	int minCDF = getMinCDF(cdfArray);
	int maxCDF = getMaxCDF(cdfArray);

	//Normalizamos el cdf
	for (int i = 0; i < 256; i++) {
		cdfArray_N[i] = (cdfArray[i]-minCDF) * 255 / (maxCDF-minCDF);
	}

	return cdfArray_N;
}
int getFrequencyOf(int value, Mat imagen) {
	//Determinamos las dimensiones de la imagen
	int filas = imagen.rows;
	int columnas = imagen.cols;

	//Inicializamos la frecuencia de cada valor en 0
	int frecuencia = 0;

	for (int i = 0; i < filas; i++) {
		for (int j = 0; j < columnas; j++) {
			int pixelValue = imagen.at<uchar>(Point(i, j));
			if (pixelValue == value) frecuencia++;
		}
	}
	return frecuencia;
}
int getMinCDF(int* cdfArray) {
	int min = cdfArray[0];
	//Obtenemos el valor minimo del array llenado
	for (int i = 1; i < 256; i++) {
		if (cdfArray[i] < min) min = cdfArray[i];
	}
	return min;
}
int getMaxCDF(int* cdfArray) {
	int max = cdfArray[0];
	//Obtenemos el valor maximo del array llenado
	for (int i = 1; i < 256; i++) {
		if (cdfArray[i] > max) max = cdfArray[i];
	}
	return max;
}

//Filtro Sobel
double** getMascXSobel(void){
	double** mascx = new double* [3];
	for (int i = 0; i < 3; i++) {
		mascx[i] = new double[3];
	}

	mascx[0][0] = -1;
	mascx[0][1] = 0;
	mascx[0][2] = 1;

	mascx[1][0] = -2;
	mascx[1][1] = 0;
	mascx[1][2] = 2;

	mascx[2][0] = -1;
	mascx[2][1] = 0;
	mascx[2][2] = 1;

	return mascx;
}
double** getMascYSobel(void){
	double** mascy = new double* [3];
	for (int i = 0; i < 3; i++) {
		mascy[i] = new double[3];
	}

	mascy[0][0] = -1;
	mascy[0][1] = -2;
	mascy[0][2] = -1;

	mascy[1][0] = 0;
	mascy[1][1] = 0;
	mascy[1][2] = 0;

	mascy[2][0] = 1;
	mascy[2][1] = 2;
	mascy[2][2] = 1;

	return mascy;
}
Mat setMascaraSobel(Mat imagen, double** mascara) {
	//Primero redimensionamos la imagen
	Mat imgRedim = redimensionarGray(imagen, imagen.rows, imagen.cols, 3);
	//Imprimimos la imagen redimensionada
	char nombre[] = "Imagen redim";
	mostrarImagen(nombre, imgRedim);
	//Aplicamos la convolucion
	Mat imagenFiltrada(imagen.rows, imagen.cols, CV_8UC1);

	for (int i = 1; i < imgRedim.rows - 1; i++) {
		for (int j = 1; j < imgRedim.cols - 1; j++) {
			imagenFiltrada.at<uchar>(Point(i - 1, j - 1)) = uchar(convolucionSobel(i, j, mascara, imgRedim));
		}
	}

	return imagenFiltrada;
}
Mat filtroSobel(Mat imagen) {
	//Primero determinamos las dimensiones de la imagen
	int filas = imagen.rows;
	int columnas = imagen.cols;

	//Creamos dos imagenes nuevas para X y Y
	Mat imagenX;
	Mat imagenY;

	//Obtenemos las imagenes filtradas por las mascaras
	double** mascaraX = getMascXSobel();
	double** mascaraY = getMascYSobel();

	//imprimirMatriz(mascaraX, 3);
	//imprimirMatriz(mascaraY, 3);

	imagenX = setMascaraSobel(imagen, mascaraX);
	imagenY = setMascaraSobel(imagen, mascaraY);

	//Obtenemos ahora la imagen final con sobel
	Mat imgSobel = sobelFunction(imagenX, imagenY, 200);

	double** matAngulos = angulos(imagenX, imagenY);
	Mat nsm = non_MS(imgSobel, matAngulos);

	Mat hister1 = primerUmbralado(nsm, 30, 120);

	//Imprimimos las imagenes de transicion
	char nombreX[] = "Imagen Sobel X";
	mostrarImagen(nombreX, imagenX);
	char nombreY[] = "Imagen Sobel Y";
	mostrarImagen(nombreY, imagenY);
	char nombreImgSobel[] = "Imagen_Sobel.png";
	mostrarImagen(nombreImgSobel, imgSobel);
	char nombreCanny[] = "Imagen Canny";
	mostrarImagen(nombreCanny, nsm);
	char nombreH1[] = "Hister1";
	mostrarImagen(nombreH1, hister1);

	return imgSobel;
}
Mat sobelFunction(Mat X, Mat Y, int umbral) {
	//Determinamos las dimensiones de la imagen de salida
	int filas = X.rows;
	int columnas = Y.cols;

	//Declaramos nuestra imagen de salida
	Mat imgSalida(filas, columnas, CV_8UC1);

	//Establecemos el umbral
	int v_X, v_Y;
	double G;
	//Recorremos la imagen de salida para ir asignando los valores
	for (int i = 0; i < filas; i++) {
		for (int j = 0; j < columnas; j++) {
			v_X = X.at<uchar>(Point(i, j));
			v_Y = Y.at<uchar>(Point(i, j));

			//G = sqrt(pow(v_X, 2) + pow(v_Y, 2));
			G = abs(v_X) + abs(v_Y);

			//Obtenemos el valor redondeado de G
			G = static_cast<int>(G);

			//Por ultimo asignamos el valor a la imagen de salida
			imgSalida.at<uchar>(Point(i, j)) = uchar(G);
		}
	}
	return imgSalida;
}
int convolucionSobel(int fila, int columna, double** mascara, Mat imagen) {
	int pixel = 0;
	double valor = 0;
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			pixel = imagen.at<uchar>(Point(fila + i, columna + j));
			valor += pixel * mascara[i + 1][j + 1];
		}
	}
	return abs(valor);
}

//Filtro Gaussiano
Mat implementarGauss(Mat imagen, int filas, int columnas) {
	int N; //Dimensiones de la mascara
	double sigma; //Valor de sigma
	double** mascara; //Kernel del filtro

	/******************CREACION Y DECLARACION DE LA MASCARA DE GAUSS**************/
	//Solicitamos las dimensiones del kernel al usuario
	cout << "Ingrese el valor para N x N del Kernel" << endl;
	cin >> N;
	//Solicitamos el valor de sigma
	cout << "Ingrese el valor para sigma de la mascara" << endl;
	cin >> sigma;

	//Generamos la matriz donde se almacena mascara
	mascara = generarMascaraGauss(N, sigma);
	imprimirMatriz(mascara, N);
	double sumaKernel = getSumaKernel(mascara, N);
	/******************************************************************************/

	/*********Creamos o redimensionamos la imagen y guardamos ese valor en otra imagen***/
	Mat nuevaImagen = imagenRedim(imagen, filas, columnas, N);
	int filasNueva = nuevaImagen.rows;
	int columnasNueva = nuevaImagen.cols;

	//Mostramos la nueva imagen redimensionada al usuario
	char nombreVentanaRedim[] = "Imagen Redim Gauss";
	mostrarImagen(nombreVentanaRedim, nuevaImagen);
	/************************************************************************************/

	Mat imagenGauss = aplicarFiltro(nuevaImagen, mascara, filasNueva, columnasNueva, N, filas, columnas, sumaKernel);
	char nombreImgFiltroGauss[] = "Imagen_Gaussiana.xls";
	mostrarImagen(nombreImgFiltroGauss, imagenGauss);

	return imagenGauss;
}
double** generarMascaraGauss(int N, double sigma) {
	double** mascara = new double* [N];
	for (int i = 0; i < N; i++) {
		mascara[i] = new double[N];
		for (int j = 0; j < N; j++) {
			int posX = i - (N / 2);
			int posY = j - (N / 2);

			mascara[i][j] = calcularGauss(posX, posY, sigma);
		}
	}
	return mascara;
}
double calcularGauss(int x, int y, double sigma) {
	double gaussValue = 0.0;
	double pi = 3.1416;

	gaussValue = (1 / (2 * pi * pow(sigma, 2))) * exp((-1) * ((pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2))));
	return gaussValue;
}
double getSumaKernel(double** kernel, int N) {
	double suma = 0.0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			suma += kernel[i][j];
		}
	}
	return suma;
}

//Deteccion de bordes Canny
double** angulos(Mat imgX, Mat imgY) {
	int filas = imgX.rows;
	int columnas = imgY.cols;
	double** matriz = new double* [filas];
	double cociente;
	int v_X, v_Y;
	for (int i = 0; i < filas; i++) {
		matriz[i] = new double[columnas];
	}
	for (int i = 0; i < filas; i++) {
		for (int j = 0; j < columnas; j++) {
			v_X = imgX.at<uchar>(Point(i, j));
			v_Y = imgY.at<uchar>(Point(i, j));
			cociente = static_cast<double>(v_Y) / static_cast<double>(v_X);
			cociente = atan(cociente);
			//Pasamos de radianes a grados
			cociente = (cociente * 180) / 3.1416;
			matriz[i][j] = cociente;
		}
	}
	//imprimirMatriz(matriz, filas);
	return matriz;
}
Mat non_MS(Mat magnitudes, double** angulos) {
	//Primero obtenemos las dimensiones
	int filas = magnitudes.rows;
	int columnas = magnitudes.cols;

	//imprimirMatriz(angulos, filas);

	//Redimensionamos la imagen
	Mat imgRedim = redimensionarGray(magnitudes, filas, columnas, 3);

	//Declaramos la matriz de salida
	Mat nms(filas, columnas, CV_8UC1);

	//Declaramos variables
	int direccion, valorP, valorL, valorR;

	//Ahora recorremos y hacemos la comparacion
	for (int i = 1; i < filas + 1; i++) {
		for (int j = 1; j < columnas + 1; j++) {
			//Obtenemos la direccion en base al angulo
			direccion = getDirByArg(angulos[i-1][j-1]);
			//cout << direccion << endl;
			//waitKey(0);
			switch (direccion){
			case 1: 
				valorP = imgRedim.at<uchar>(Point(i, j));
				valorL = imgRedim.at<uchar>(Point(i, j - 1));
				valorR = imgRedim.at<uchar>(Point(i, j + 1));
				if (valorP >= valorL && valorP >= valorR) nms.at<uchar>(Point(i-1, j-1)) = uchar(valorP);
				else nms.at<uchar>(Point(i-1, j-1)) = uchar(0);
				break;
			case 2:
				valorP = imgRedim.at<uchar>(Point(i,j));
				valorL = imgRedim.at<uchar>(Point(i+1, j-1));
				valorR = imgRedim.at<uchar>(Point(i-1, j +1));
				if (valorP >= valorL && valorP >= valorR) nms.at<uchar>(Point(i-1, j-1)) = uchar(valorP);
				else nms.at<uchar>(Point(i - 1, j - 1)) = uchar(0);
				break;
			case 3:
				valorP = imgRedim.at<uchar>(Point(i, j));
				valorL = imgRedim.at<uchar>(Point(i-1, j));
				valorR = imgRedim.at<uchar>(Point(i+1, j));
				if (valorP >= valorL && valorP >= valorR) nms.at<uchar>(Point(i-1, j-1)) = uchar(valorP);
				else nms.at<uchar>(Point(i - 1, j - 1)) = uchar(0);
				break;
			case 4:
				valorP = imgRedim.at<uchar>(Point(i, j));
				valorL = imgRedim.at<uchar>(Point(i-1, j-1));
				valorR = imgRedim.at<uchar>(Point(i + 1, j + 1));
				if (valorP >= valorL && valorP >= valorR) nms.at<uchar>(Point(i-1, j-1)) = uchar(valorP);
				else nms.at<uchar>(Point(i - 1, j - 1)) = uchar(0);
				break;
			}
		}
	}
	return nms;
}
int getDirByArg(double angulo) {
	int direccion = 0;
	if ((angulo >= 0 && angulo < 22.5) || (angulo >= 180 && angulo < 202.5) || (angulo >= 337.5 && angulo < 360) || (angulo >= 157.5 && angulo < 180)) { 
		direccion = 1; 
	}else if ((angulo >= 22.5 && angulo < 67.5) || (angulo >= 202.5 && angulo < 247.5)) {
		direccion = 2; 
	}else if ((angulo >= 67.5 && angulo < 112.5) || (angulo >= 247.5 && angulo < 292.5)) {
		direccion = 3; 
	}else if ((angulo >= 112.5 && angulo < 157.5) || (angulo >= 247.5 && angulo < 337.5)) {
		direccion = 4; 
	}
	return direccion;
}
Mat primerUmbralado(Mat imagenNSM, int lowValue, int highValue) {
	int filas = imagenNSM.rows;
	int columnas = imagenNSM.cols;
	int valueAt;

	Mat imgSalida(filas, columnas, CV_8UC1);

	for (int i = 0; i < filas; i++) {
		for (int j = 0; j < columnas; j++) {
			valueAt = imagenNSM.at<uchar>(Point(i, j));
			if (valueAt >= highValue) imgSalida.at<uchar>(Point(i, j)) = uchar(255);
			else if (valueAt <= lowValue) imgSalida.at<uchar>(Point(i, j)) = uchar(0);
			//else if(isBorder(imagenNSM, i, j) == 1) imgSalida.at<uchar>(Point(i, j)) = uchar(255);
			else imgSalida.at<uchar>(Point(i, j)) = uchar(0);
		}
	}
	return imgSalida;
}
int isBorder(Mat imagen, int x, int y) {
	int isIt = 0;

	//Detectamos si el pixel puede ser borde o no puede ser borde de la imagen en base a su posiclion con respecto a otros bordes

	if (y >= 1) {
		if (imagen.at<uchar>(Point(x - 1, y - 1)) >= 60) isIt = 1;
		if (imagen.at<uchar>(Point(x, y - 1)) >= 60) isIt = 1;
		if (imagen.at<uchar>(Point(x + 1, y - 1)) >= 60) isIt = 1;
	}
	if (y < imagen.cols-1) {
		if (imagen.at<uchar>(Point(x - 1, y + 1)) >= 60) isIt = 1;
		if (imagen.at<uchar>(Point(x, y + 1)) >= 60) isIt = 1;
		if (imagen.at<uchar>(Point(x + 1, y -+1)) >= 60) isIt = 1;
	}
	if (x >= 1) {
		if (imagen.at<uchar>(Point(x - 1, y - 1)) >= 60) isIt = 1;
		if (imagen.at<uchar>(Point(x - 1, y)) >= 60) isIt = 1;
		if (imagen.at<uchar>(Point(x - 1, y + 1)) >= 60) isIt = 1;
	}
	if (x < imagen.rows-1) {
		if (imagen.at<uchar>(Point(x + 1, y - 1)) >= 60) isIt = 1;
		if (imagen.at<uchar>(Point(x + 1, y)) >= 60) isIt = 1;
		if (imagen.at<uchar>(Point(x + 1, y + 1)) >= 60) isIt = 1;
	}

	return isIt;
}

//Procesamientos
Mat imagenRedim(Mat imagen, int filas, int columnas, int N) {
	int nuevaFilas = filas + (N - 1);
	int nuevaColumnas = columnas + (N - 1);
	int azul, verde, rojo;

	Mat nueva_imagen(nuevaFilas, nuevaColumnas, CV_8UC1);

	for (int i = 0; i < nuevaFilas; i++) {
		for (int j = 0; j < nuevaColumnas; j++) {
			if (i <= ((N / 2) - 1) || j <= ((N / 2) - 1) || i >= ((N / 2)) + filas || j >= ((N / 2)) + columnas) {
				nueva_imagen.at<uchar>(Point(j, i)) = uchar(0);
			}
			else {
				azul = imagen.at<Vec3b>(Point(j, i)).val[0];
				verde = imagen.at<Vec3b>(Point(j, i)).val[1];
				rojo = imagen.at<Vec3b>(Point(j, i)).val[2];

				int gris_v = (azul * 0.114 + verde * 0.587 + rojo * 0.299);
				nueva_imagen.at<uchar>(Point(j, i)) = uchar(gris_v);
			}
		}
	}
	return nueva_imagen;
}
Mat redimensionarGray(Mat imagen, int filas, int columnas, int N) {
	int nuevaFilas = filas + (N - 1);
	int nuevaColumnas = columnas + (N - 1);

	Mat nueva_imagen(nuevaFilas, nuevaColumnas, CV_8UC1);

	for (int i = 0; i < nuevaFilas; i++) {
		for (int j = 0; j < nuevaColumnas; j++) {
			if (i <= ((N / 2) - 1) || j <= ((N / 2) - 1) || i >= ((N / 2)) + filas || j >= ((N / 2)) + columnas) {
				nueva_imagen.at<uchar>(Point(j, i)) = uchar(0);
			}
			else {
				int gris_v = imagen.at<uchar>(Point(j, i));
				nueva_imagen.at<uchar>(Point(j, i)) = uchar(gris_v);
			}
		}
	}
	return nueva_imagen;
}
Mat aplicarFiltro(Mat imagen, double** mascara, int imgX, int imgY, int N, int dimImgX, int dimImgY, double suma) {
	Mat imagenFiltrada(dimImgX, dimImgY, CV_8UC1); //Declaramos la imagen de destino
	int expansionAplicada = (N-1) / 2; //Calculamos cuanto ampliamos a cada lado

	for (int i = 0; i < imgX; i++) {
		if ((i >= expansionAplicada) && (i < imgX - expansionAplicada)) {
			for (int j = 0; j < imgY; j++) {
				if ((j >= expansionAplicada) && (j < imgY - expansionAplicada)) {
					double valor = 0.0;
					int marcadorX = expansionAplicada * (-1), marcadorY;
					for (int k = 0; k < N; k++) {
						marcadorY = expansionAplicada * (-1);
						for (int l = 0; l < N; l++) {
							valor += mascara[k][l] * (double)imagen.at<uchar>(Point(i + marcadorX, j + marcadorY));
							marcadorY++;
						}
						marcadorX++;
					}
					//Ahora asignamos el valor
					int val = valor / suma;
					imagenFiltrada.at<uchar>(Point(i - expansionAplicada, j - expansionAplicada)) = uchar(val);
				}
			}
		}
	}
	return imagenFiltrada;
}


//Operadores
void imprimirMatriz(double** mascara, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << "[" << mascara[i][j] << "]";
		}
		cout << endl;
	}
	return;
}
void mostrarImagen(char nombreVentana[], Mat imagen) {
	//Determinamos las dimensiones de la imagen para mostrarlas tambien
	int filas = imagen.rows;
	int columnas = imagen.cols;

	//Mostramos la informacion de la imagen
	cout << "Dimensiones de la Imagen "<<nombreVentana<< endl;
	cout << "Filas: " << filas << endl;
	cout << "Columnas: " << columnas << endl;
	
	//Mostramos la imagen
	namedWindow(nombreVentana, WINDOW_AUTOSIZE);
	imshow(nombreVentana, imagen);
	waitKey(0);
	return;
}
