#include "ocv_utils.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <vector>
#include <map>

using namespace cv;
using namespace std;

// ------------------------------------------------------------------------------------------------------------------------

void printHelp(const string &progName)
{
    cout << "Usage:\n\t " << progName << " <image_file> <K_num_of_clusters> [<image_ground_truth>]" << endl;
}

int min(int a, int b)
{
    return a < b ? a : b;
}

int max(int a, int b)
{
    return a > b ? a : b;
}

/**
 * @brief Calculate the euclidean distance between {r,g,b} and {R,G,B}
 *
 * @param r red
 * @param g green
 * @param b blue
 * @param R Red
 * @param G Green
 * @param B Blue
 * @return distance between {r,g,b} and {R,G,B}
 */
int distance(int r, int g, int b, int R, int G, int B)
{
    return sqrt((r - R) * (r - R) + (g - G) * (g - G) + (b - B) * (b - B));
}

/**
 * @brief Calculate the index of the closest center from the color
 *
 * @param color Color of the pixel
 * @param centers All the centers
 * @return index of the closest center
 */
int closestCenter(Vec3b color, Mat centers)
{
    // Extract r g b values
    int r = color[0];
    int g = color[1];
    int b = color[2];

    // Initialize variables
    int minDistance = sqrt(3 * 255 * 255) + 1;
    int closest;

    // Test each centers
    for (int i = 0; i < centers.rows; i++)
    {
        // Extract R G B values from the current center
        int R = centers.at<float>(i, 0);
        int G = centers.at<float>(i, 1);
        int B = centers.at<float>(i, 2);

        // Replace varibales with new values if necessary
        int curDistance = distance(r, g, b, R, G, B);
        if (curDistance < minDistance)
        {
            minDistance = curDistance;
            closest = i;
        }
    }
    return closest;
}

// ------------------------------------------------------------------------------------------------------------------------

/**
 * @brief Calculate the k-means of the vect
 *
 * @param vect
 * @param k
 * @return Centers of vect
 */
Mat kmeans(Mat sourceMatrix, Mat vect, int k)
{
    Mat centers(k, 1, CV_32FC3);

    // initialisation des centres
    for (int i = 0; i < k; i++)
    {
        centers.at<Vec3f>(i) = vect.at<Vec3f>(i * vect.rows / k);
    }

    int it = 0;
    int it_max = 20000;
    int counters[k];
    Mat prevCenters;

    do
    {
        // initialiser les varibales de boucle
        prevCenters = centers.clone();
        for (int i = 0; i < k; i++)
        {
            counters[i] = 0;
        }

        // compter les centres
        // sommer les pixels de chaque classe
        centers = Mat::zeros(k, 1, CV_32FC3);
        for (int i = 0; i < vect.cols; i++)
        {
            Vec3f color = vect.at<Vec3f>(i);
            int closest = closestCenter(color, prevCenters);
            centers.at<Vec3f>(closest) += color;
            counters[closest]++;
        }

        // faire la moyennes des pixels de chaque classe
        for (int i = 0; i < k; i++)
        {
            if (counters[i] != 0)
                centers.at<Vec3f>(i) /= counters[i];
        }


        Mat resultMatrix = sourceMatrix.clone();
        // Loop for each pixel
        for (int i = 0; i < resultMatrix.rows; i++)
        {
            for (int j = 0; j < resultMatrix.cols; j++)
            {
                Vec3b color = sourceMatrix.at<Vec3b>(i, j);

                int closest = closestCenter(color, centers);

                // Set pixel color to the color of the closest center
                resultMatrix.at<Vec3b>(i, j) = centers.at<Vec3b>(closest);
            }
        }
        
        imshow("kmeans result", resultMatrix);
        waitKey(1000);  // Attendre un peu pour la mise à jour de l'affichage

    } while (it++ < it_max && sum(centers != prevCenters) != Scalar(0, 0, 0, 0));
    cout << "kmeans terminé en " << it << " itérations." << endl << endl;

    return centers;
}

// ------------------------------------------------------------------------------------------------------------------------
// Comparateur personnalisé pour Vec3f
struct Vec3fCompare {
    bool operator()(const Vec3f& a, const Vec3f& b) const {
        if (a[0] != b[0])
            return a[0] < b[0];
        if (a[1] != b[1])
            return a[1] < b[1];
        return a[2] < b[2];
    }
};


/**
* @brief Calcule la distance chromatique entre deux pixels et met à jour les matrices de moyennes et de comptages.
 * 
 * @param s_i Indice de la ligne du premier pixel.
 * @param s_j Indice de la colonne du premier pixel.
 * @param d_i Indice de la ligne du deuxième pixel.
 * @param d_j Indice de la colonne du deuxième pixel.
 * @param matIMG Pointeur vers la matrice de l'image contenant les valeurs des pixels.
 * @param matMOY Pointeur vers la matrice contenant les valeurs moyennes des pixels.
 * @param matCOUNT Pointeur vers la matrice contenant les comptages des pixels.
 * @param hc Seuil chromatique pour la distance.
 * @return float La distance chromatique entre les deux pixels si elle est inférieure au seuil, sinon 0.
 */
float compute(int s_i, int s_j, int d_i, int d_j, Mat *matIMG, Mat *matMOY, Mat *matCOUNT, double hc)
{
    Vec3f centre = matIMG->at<Vec3f>(s_i, s_j);
    Vec3f dest = matIMG->at<Vec3f>(d_i, d_j);

    // seuil chromatique
    float dist = cv::norm(dest - centre);
    if (dist < hc)
    {
        matMOY->at<Vec3f>(s_i, s_j) += dest;
        matMOY->at<Vec3f>(d_i, d_j) += centre;

        matCOUNT->at<float>(s_i, s_j)++;
        matCOUNT->at<float>(d_i, d_j)++;

        return dist;
    }

    return 0;
}

/**
 * @brief Vérifie si la norme euclidienne de tous les éléments de la matrice (M1 - M2) est inférieure à epsilon.
 * 
 * @param M1 Première matrice.
 * @param M2 Deuxième matrice.
 * @param epsilon Seuil de tolérance.
 * @return false Si tous les éléments de la matrice de différence sont inférieurs à epsilon.
 * @return true Sinon.
 */
bool isDifferenceBelowEpsilon(const Mat M1, const Mat M2, double epsilon)
{
    // Vérifie que les matrices ont les mêmes dimensions et le même type
    if (M1.size() != M2.size() || M1.type() != M2.type())
    {
        cerr << "Les matrices doivent avoir les mêmes dimensions et le même type." << endl;
        return false;
    }

    // Calcule la différence entre les deux matrices
    Mat diff;
    absdiff(M1, M2, diff);

    // Vérifie que la norme euclidienne de tous les éléments de la matrice de différence est inférieure à epsilon
    for (int i = 0; i < diff.rows; ++i)
    {
        for (int j = 0; j < diff.cols; ++j)
        {
            // Pour les matrices de type CV_32FC3 (ou CV_64FC3), calculez la norme euclidienne pour chaque canal
            if (diff.type() == CV_32FC3 || diff.type() == CV_64FC3)
            {
                Vec3f pixel = diff.at<Vec3f>(i, j);
                float norm = sqrt(pixel[0] * pixel[0] + pixel[1] * pixel[1] + pixel[2] * pixel[2]);
                if (norm > epsilon)
                {
                    return true;
                }
            }
            // Pour les autres types, vérifiez directement la valeur
            else
            {
                float value = diff.at<float>(i, j);
                if (fabs(value) > epsilon)
                {
                    return true;
                }
            }
        }
    }

    return false;
}


/**
 * @brief Calcule les clusters de pixels en utilisant l'algorithme mean shift.
 * 
 * @param hs Seuil spatiale.
 * @param hc Seuil chromatique.
 * @param eps Tolérance pour la convergence (non utilisé dans cette implémentation).
 * @param itMax Nombre maximal d'itérations.
 * @param matIMG Matrice de l'image contenant les valeurs des pixels.
 * @return Mat Matrice de l'image segmentée après application de l'algorithme mean shift.
 */
Mat meanshift(uint hs, double hc, double eps, uint itMax, Mat matIMG)
{
    Mat matMOY, matCOUNT;
    Mat prev_matIMG=matIMG.clone();
    float maxDist, dist;
    int i, j;
    
    int it = 0;

    #pragma omp parallel shared(it)

    do {
        #pragma omp single
        {
        cout << " mean-shit (" << it << "/" << itMax << ")" << endl;

        matMOY = Mat::zeros(matIMG.rows, matIMG.cols, CV_32FC3);
        matCOUNT = Mat::zeros(matIMG.rows, matIMG.cols, CV_32FC1);
        maxDist = 0;
        it++;
        }

        #pragma omp for private(i, j)
        for (i = 0; i < matIMG.rows; i++)
        {
            for (j = 0; j < matIMG.cols; j++)
            {
                // pixels à droite du centre
                for (int j2 = j; j2 < min(j + hs + 1, matIMG.cols); j2++)
                {
                    dist = compute(i, j, i, j2, &matIMG, &matMOY, &matCOUNT, hc);
                    if (dist > maxDist)
                    {
                        maxDist = dist;
                    }
                }

                // pixels en dessous du centre
                for (int i2 = i + 1; i2 < min(i + hs + 1, matIMG.rows); i2++)
                {
                    for (int j2 = max(j - hs, 0); j2 < min(j + hs + 1, matIMG.cols); j2++)
                    {
                        dist = compute(i, j, i2, j2, &matIMG, &matMOY, &matCOUNT, hc);
                        if (dist > maxDist)
                        {
                            maxDist = dist;
                        }
                    }
                }
                matIMG.at<Vec3f>(i, j) = matMOY.at<Vec3f>(i, j) / matCOUNT.at<float>(i, j);
            }
        }
        // cout << "maxDist = " << maxDist << endl;
        // Affichage de l'image après chaque itération
        Mat msres;
        matIMG.convertTo(msres, CV_8U);
        imshow("Mean-shift result", msres);
        waitKey(1);  // Attendre un peu pour la mise à jour de l'affichage

    }
    while (it < itMax && isDifferenceBelowEpsilon(prev_matIMG, matIMG, eps));
    cout << "meanshift terminé en " << it << " itérations." << endl << endl;
    return matIMG;
}

/**
 * @brief Détecte les modes (maximums locaux) dans l'image segmentée.
 * 
 * @param matIMG Matrice de l'image segmentée après application de l'algorithme mean shift.
 * @param eps Tolérance pour la convergence.
 * @return vector<Vec3f> Liste des modes détectés.
 */
vector<Vec3f> detectModes(const Mat& matIMG, double eps)
{
    map<Vec3f, int, Vec3fCompare> modeMap;  // Map pour stocker les modes et leur comptage
    vector<Vec3f> modes;

    // Utilisation de OpenMP pour paralléliser la boucle sur les pixels de l'image
    #pragma omp parallel for shared(modeMap, modes)
    for (int i = 0; i < matIMG.rows; ++i)
    {
        for (int j = 0; j < matIMG.cols; ++j)
        {
            Vec3f pixel = matIMG.at<Vec3f>(i, j);
            bool found = false;

            // Cherche si ce pixel est proche d'un mode existant
            #pragma omp critical
            {
                for (const auto& mode : modes)
                {
                    if (cv::norm(mode - pixel) < eps)
                    {
                        modeMap[mode]++;
                        found = true;
                        break;
                    }
                }

                // Si aucun mode proche n'est trouvé, ajoute ce pixel comme un nouveau mode
                if (!found)
                {
                    modes.push_back(pixel);
                    modeMap[pixel] = 1;
                }
            }
        }
    }

    // Trie les modes par leur comptage (optionnel)
    sort(modes.begin(), modes.end(), [&modeMap](const Vec3f& a, const Vec3f& b) {
        return modeMap[a] > modeMap[b];
    });

    return modes;
}

/**
 * @brief Attribue des labels aux pixels en fonction des modes détectés.
 * 
 * @param matIMG Matrice de l'image segmentée après application de l'algorithme mean shift.
 * @param modes Liste des modes détectés.
 * @param eps Tolérance pour la convergence.
 * @return Mat Matrice contenant les labels des pixels.
 */
Mat labelPixels(const Mat& matIMG, const vector<Vec3f>& modes)
{
    Mat labeledImage = matIMG.clone(); // Crée une copie modifiable de matIMG

    // Utilisation de OpenMP pour paralléliser la boucle sur les pixels de l'image
    #pragma omp parallel for
    for (int i = 0; i < matIMG.rows; ++i)
    {
        for (int j = 0; j < matIMG.cols; ++j)
        {
            Vec3f pixel = matIMG.at<Vec3f>(i, j);
            int label = -1;
            double minDist = DBL_MAX;

            // Trouve le mode le plus proche pour ce pixel
            for (int k = 0; k < modes.size(); ++k)
            {
                double dist = cv::norm(pixel - modes[k]);
                if (dist < minDist)
                {
                    minDist = dist;
                    label = k;
                }
            }

            Vec3f mode_value = modes[label]; // Copie la valeur du mode dans une variable modifiable
            labeledImage.at<Vec3f>(i, j) = mode_value; // Affecte la valeur à labeledImage
        }
    }

    return labeledImage; // Retourne la copie modifiée de matIMG
}



// ------------------------------------------------------------------------------------------------------------------------
/**
 * @brief Calculate the euclidean distance between two 5-channel vectors
 *
 * @param v1 First 5-channel vector
 * @param v2 Second 5-channel vector
 * @return distance between v1 and v2
 */
int distance5(const Vec<float, 5>& a, const Vec<float, 5>& b)
{
    return sqrt((a[0] - b[0]) * (a[0] - b[0]) +
                (a[1] - b[1]) * (a[1] - b[1]) +
                (a[2] - b[2]) * (a[2] - b[2]) +
                (a[3] - b[3]) * (a[3] - b[3]) +
                (a[4] - b[4]) * (a[4] - b[4]));
}

/**
 * @brief Calculate the index of the closest center from the vector
 *
 * @param vec 5-channel vector
 * @param centers All the centers
 * @return index of the closest center
 */
int closestCenter5(const Vec<float, 5>& sample, const vector<Vec<float, 5>>& centers)
{
    int minDistance = std::numeric_limits<int>::max();
    int closest = -1;

    for (int i = 0; i < centers.size(); i++)
    {
        int curDistance = distance5(sample, centers[i]);
        if (curDistance < minDistance)
        {
            minDistance = curDistance;
            closest = i;
        }
    }
    return closest;
}


/**
 * @brief Calculate the k-means of the vect
 *
 * @param vect
 * @param k
 * @return Centers of vect
 */
vector<Vec<float, 5>> kmeans5(const vector<Vec<float, 5>>& samples, int k)
{
    vector<Vec<float, 5>> centers(k);

    // Initialisation des centres
    for (int i = 0; i < k; i++)
    {
        centers[i] = samples[i * samples.size() / k];
    }

    int it = 0;
    const int it_max = 10;
    vector<int> counters(k, 0);
    vector<Vec<float, 5>> prevCenters(k);

    do
    {
        // Initialiser les variables de boucle
        prevCenters = centers;
        fill(counters.begin(), counters.end(), 0);
        fill(centers.begin(), centers.end(), Vec<float, 5>(0, 0, 0, 0, 0));

        // Compter les centres et sommer les pixels de chaque classe
        #pragma omp parallel for
        for (int i = 0; i < samples.size(); i++)
        {
            int closest = closestCenter5(samples[i], prevCenters);
            #pragma omp critical
            {
                centers[closest] += samples[i];
                counters[closest]++;
            }
        }

        // Faire la moyennes des pixels de chaque classe
        for (int i = 0; i < k; i++)
        {
            if (counters[i] != 0)
                centers[i] /= counters[i];
        }

    } while (it++ < it_max && !equal(centers.begin(), centers.end(), prevCenters.begin(),
                                     [](const Vec<float, 5>& a, const Vec<float, 5>& b) {
                                         return norm(a - b) < 1e-4; // Tolérance pour convergence
                                     }));

    cout << "kmeans terminé en " << it << " itérations." << endl;
    return centers;
}

/**
 * @brief Calculate the superpixels of the matIMG
 * 
 * @param img 
 * @param k 
 * @param m 
 * @param itMax 
 * @param labels
 * @param centers 
 */
void superpixel(const Mat &matIMG, int k, int m, int itMax, Mat &labels, Mat &centers) {
    // Convertir l'image en espace de couleur CIE Lab
    Mat imgLab;
    cvtColor(matIMG, imgLab, COLOR_BGR2Lab);

    // Redimensionner l'image pour obtenir une liste de vecteurs à 5 dimensions (x, y, L, a, b)
    int numPixels = imgLab.rows * imgLab.cols;
    vector<Vec<float, 5>> samples(numPixels);
    for (int y = 0; y < imgLab.rows; y++) {
        for (int x = 0; x < imgLab.cols; x++) {
            Vec3b color = imgLab.at<Vec3b>(y, x);
            Vec<float, 5> sample;
            sample[0] = x * m / sqrt(k);
            sample[1] = y * m / sqrt(k);
            sample[2] = color[0];
            sample[3] = color[1];
            sample[4] = color[2];
            samples[y * imgLab.cols + x] = sample;
        }
    }

    // Exécuter le clustering k-means
    vector<Vec<float, 5>> centers5 = kmeans5(samples, k);

    // Convertir les positions des centres
    for (auto &center : centers5) {
        center[0] *= sqrt(k) / m;
        center[1] *= sqrt(k) / m;
    }

    // Convertir les centres en un format plus pratique (Vec3b)
    centers = Mat(k, 1, CV_8UC3);
    for (int i = 0; i < k; i++) {
        centers.at<Vec3b>(i, 0) = Vec3b(centers5[i][2], centers5[i][3], centers5[i][4]);
    }

    // Attribuer des étiquettes à chaque pixel
    labels = Mat(imgLab.rows, imgLab.cols, CV_32SC1);
    for (int y = 0; y < imgLab.rows; y++) {
        for (int x = 0; x < imgLab.cols; x++) {
            Vec<float, 5> sample;
            sample[0] = x * m / sqrt(k);
            sample[1] = y * m / sqrt(k);
            sample[2] = imgLab.at<Vec3b>(y, x)[0];
            sample[3] = imgLab.at<Vec3b>(y, x)[1];
            sample[4] = imgLab.at<Vec3b>(y, x)[2];
            labels.at<int>(y, x) = closestCenter5(sample, centers5);
        }
    }

    // Créer l'image segmentée finale
    Mat result(matIMG.size(), matIMG.type());
    for (int y = 0; y < matIMG.rows; y++) {
        for (int x = 0; x < matIMG.cols; x++) {
            int label = labels.at<int>(y, x);
            result.at<Vec3b>(y, x) = centers.at<Vec3b>(label, 0);
        }
    }

    // Convertir l'image segmentée de Lab à BGR
    cvtColor(result, result, COLOR_Lab2BGR);

    // Afficher le résultat
    imshow("Superpixel Segmentation", result);
    waitKey(0);
}

// ------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char **argv)
{
    if (argc != 3 && argc != 4)
    {
        cout << " Incorrect number of arguments." << endl;
        printHelp(string(argv[0]));
        return EXIT_FAILURE;
    }

    const auto imageFilename = string(argv[1]);
    const string groundTruthFilename = (argc == 4) ? string(argv[3]) : string();
    const int k = stoi(argv[2]);

    if (k < 1)
    {
        cout << " k must be a positive integer" << endl;
        printHelp(string(argv[0]));
        return EXIT_FAILURE;
    }

    // trust file used ?
    bool useTrust = !groundTruthFilename.empty() && k == 2;

    // just for debugging
    {
        cout << endl
             << " Program called with the following arguments:" << endl;
        cout << " \timage file: " << imageFilename << endl;
        cout << " \tk: " << k << endl;
        if (useTrust)
            cout << " \tground truth segmentation: " << groundTruthFilename << endl
                 << endl;
    }

    // load the color image to process from file
    Mat sourceMatrix = imread(imageFilename, IMREAD_COLOR);
    Mat sourceMatrix_sup;
    sourceMatrix_sup= sourceMatrix.clone();
    Mat trustMatrix = imread(groundTruthFilename, IMREAD_COLOR);

    // for debugging use the macro PRINT_MAT_INFO to print the info about the matrix, like size and type
    PRINT_MAT_INFO(sourceMatrix);

    // convert the image into floats (CV_32F), for kmeans
    Mat sourceMatrix32f;
    sourceMatrix.convertTo(sourceMatrix32f, CV_32F);
    // reshape the image into a 1D array, for kmeans
    Mat sourceMatrixReshaped = sourceMatrix32f.reshape(3, 1);

    // Call OpenCV's kmeans function
    Mat labels, centers;
    //kmeans(sourceMatrixReshaped, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    // Call ou kmeans function
    centers = kmeans(sourceMatrix, sourceMatrixReshaped, k);

    // Create result matrix
    Mat resultMatrix = sourceMatrix.clone();
    Mat diffMatrix = trustMatrix.clone();

    // Some usefull colors
    Vec3b white(255, 255, 255);
    Vec3b black(0, 0, 0);
    Vec3b red(255, 0, 0);
    Vec3b blue(0, 0, 255);

    Vec3b colorMaj, colorMin;
    if (useTrust)
    {
        // Count the number of pixel per class in the trust matrix
        int n = 0, p = 0;
        for (int i = 0; i < trustMatrix.rows; i++)
        {
            for (int j = 0; j < trustMatrix.cols; j++)
            {
                if (trustMatrix.at<Vec3b>(i, j) == white)
                {
                    n++;
                }
                else
                {
                    p++;
                }
            }
        }

        // Assign black/white to colorMaj/colorMin
        if (n > p)
        {
            colorMaj = white;
            colorMin = black;
        }
        else
        {
            colorMaj = black;
            colorMin = white;
        }

        // Count the number of pixel per class in the source matrix
        n = 0;
        p = 0;
        for (int i = 0; i < resultMatrix.rows; i++)
        {
            for (int j = 0; j < resultMatrix.cols; j++)
            {
                Vec3b color = sourceMatrix.at<Vec3b>(i, j);
                int closest = closestCenter(color, centers);
                if (closest == 0)
                {
                    n++;
                }
                else
                {
                    p++;
                }
            }
        }

        // Reorganize centers to match with the trust matrix
        if (n < p)
        {
            Vec3f tmp = centers.at<Vec3f>(0);
            centers.at<Vec3f>(0) = centers.at<Vec3f>(1);
            centers.at<Vec3f>(1) = tmp;
        }
    }

    // Quality counters
    float TP = 0;
    float FP = 0;
    float TN = 0;
    float FN = 0;

    // Loop for each pixel
    for (int i = 0; i < resultMatrix.rows; i++)
    {
        for (int j = 0; j < resultMatrix.cols; j++)
        {
            Vec3b color = sourceMatrix.at<Vec3b>(i, j);

            int closest = closestCenter(color, centers);

            // Set pixel color to the color of the closest center
            // (Or black/white if we are using a trust image)
            if (useTrust && closest == 0)
            {
                resultMatrix.at<Vec3b>(i, j) = colorMaj;
            }
            else if (useTrust && closest == 1)
            {
                resultMatrix.at<Vec3b>(i, j) = colorMin;
            }
            else
            {
                resultMatrix.at<Vec3b>(i, j) = centers.at<Vec3b>(closest);
            }

            // Increment counters for the quality evaluations
            if (useTrust)
            {
                if (trustMatrix.at<Vec3b>(i, j) != resultMatrix.at<Vec3b>(i, j))
                {
                    diffMatrix.at<Vec3b>(i, j) = trustMatrix.at<Vec3b>(i, j) == colorMaj ? red : blue;

                    if (trustMatrix.at<Vec3b>(i, j) == colorMaj)
                    {
                        // False negative
                        FN++;
                    }
                    else
                    {
                        // False positive
                        FP++;
                    }
                }
                else
                {
                    if (trustMatrix.at<Vec3b>(i, j) == colorMaj)
                    {
                        // True positive
                        TP++;
                    }
                    else
                    {
                        // True  negative
                        TN++;
                    }
                }
            }
        }
    }

    // Trust comparison results
    if (useTrust)
    {
        float P = TP / (TP + FP);
        float S = TP / (TP + FN);
        float DSC = 2 * TP / (2 * TP + FP + FN);

        cout << endl
             << "counters = " << endl
             << " TP = " << TP << endl
             << " TN = " << TN << endl
             << " FP = " << FP << endl
             << " FN = " << FN << endl
             << " Total (debug) = " << (TP + TN + FP + FN) << endl
             << endl
             << " Precision = " << P << endl
             << " Sensibility = " << S << endl
             << " DICE Similarity Coefficient = " << DSC << endl;
    }

    // compute meanshift for the image
    int hs = 5;
    int hc = 20;
    double eps = 1.0;
    //double eps2 = 0.05;
    //it 23
    int ite = 100;

    Mat msMatrix = meanshift(hs, hc, eps, ite, sourceMatrix32f);
    // Détection des modes
    //texture3 et 11 bonnes avec 80
    //tex8 => 35
    vector<Vec3f> modes = detectModes(msMatrix, 80);

    // Attribution des labels
    Mat label = labelPixels(msMatrix, modes);
    Mat msMatrixU;
    label.convertTo(msMatrixU, CV_8U);

    // Nom du fichier de sortie
    //string outputFile = "result.png";

    // Enregistrer l'image binaire résultante
    //imwrite(outputFile, msMatrixU);

    // create image windows
    namedWindow("Source", cv::WINDOW_AUTOSIZE);
    namedWindow("Result", cv::WINDOW_AUTOSIZE);
    if (useTrust)
    {
        namedWindow("Trust", cv::WINDOW_AUTOSIZE);
        namedWindow("Diff", cv::WINDOW_AUTOSIZE);
    }
    namedWindow("Mean Shift", cv::WINDOW_AUTOSIZE);

    // show images
    imshow("Source", sourceMatrix);
    imshow("Result", resultMatrix);
    if (useTrust)
    {
        imshow("Trust", trustMatrix);
        imshow("Diff", diffMatrix);
    }
    imshow("Mean Shift", msMatrixU);



    // press q to end
    //while (waitKey(0) != 113);

    // compute superpixels for the image 
    waitKey(0);
    // Initialize variables for superpixel function
    Mat labels_sup, centers_sup;
    // Paramètres
    int m = 10;
    int maxIter = 10;
    int k_sup=100;

    // Call the superpixel function
    superpixel(sourceMatrix_sup, k_sup, 10, 50, labels_sup, centers_sup);



    return EXIT_SUCCESS;
}
