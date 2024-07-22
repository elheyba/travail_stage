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
* @brief Calculate the Euclidean distance between two RGB colors.
 *
 * @param r Red component of the first color
 * @param g Green component of the first color
 * @param b Blue component of the first color
 * @param R Red component of the second color
 * @param G Green component of the second color
 * @param B Blue component of the second color
 * @return Euclidean distance between the two colors
 */
int distance(int r, int g, int b, int R, int G, int B)
{
    return sqrt((r - R) * (r - R) + (g - G) * (g - G) + (b - B) * (b - B));
}

/**
 * @brief Calculate the index of the closest center from a given color.
 *
 * @param color Color of the pixel in RGB format (Vec3b)
 * @param centers Matrix containing the center colors (in RGB format)
 * @return Index of the closest center
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
 * @brief Perform k-means clustering on the given image matrix.
 *
 * @param sourceMatrix The original image matrix.
 * @param vect Matrix containing the pixel colors (in RGB format) to cluster.
 * @param k Number of clusters.
 * @return Matrix containing the cluster centers (in RGB format).
 */
Mat kmeans(Mat sourceMatrix, Mat vect, int k)
{
    Mat centers(k, 1, CV_32FC3);

    // initialisation des centres
    for (int i = 0; i < k; i++)
    {
        centers.at<Vec3f>(i) = vect.at<Vec3f>(i * vect.rows / k);
    }
    // Affichage du type de données
    PRINT_MAT_INFO(vect);
    PRINT_MAT_INFO(sourceMatrix);



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


        //Mat resultMatrix = sourceMatrix.clone();
        Mat resultMatrix(sourceMatrix.size(), CV_8UC3);
        // Loop for each pixel
        for (int i = 0; i < resultMatrix.rows; i++)
        {
            for (int j = 0; j < resultMatrix.cols; j++)
            {
                Vec3b color = sourceMatrix.at<Vec3b>(i, j);
                int closest = closestCenter(Vec3f(color[0], color[1], color[2]), centers);
                resultMatrix.at<Vec3b>(i, j) = Vec3b(centers.at<Vec3f>(closest)[0],
                                                     centers.at<Vec3f>(closest)[1],
                                                     centers.at<Vec3f>(closest)[2]);
            }
        }

        
        
        imshow("kmeans result", resultMatrix);
        waitKey(100);  // Attendre un peu pour la mise à jour de l'affichage

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
* @brief Computes the chromatic distance between two pixels and updates the mean and count matrices.
 * 
 * @param s_i Row index of the first pixel.
 * @param s_j Column index of the first pixel.
 * @param d_i Row index of the second pixel.
 * @param d_j Column index of the second pixel.
 * @param matIMG Pointer to the image matrix containing pixel values.
 * @param matMOY Pointer to the matrix containing mean pixel values.
 * @param matCOUNT Pointer to the matrix containing pixel counts.
 * @param hc Chromatic threshold for distance.
 * @return float Chromatic distance between the two pixels if below the threshold, otherwise 0.
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
 * @brief Checks if the Euclidean norm of all elements of the matrix (M1 - M2) is below epsilon.
 * 
 * @param M1 First matrix.
 * @param M2 Second matrix.
 * @param epsilon Tolerance threshold.
 * @return true If any element in the difference matrix exceeds epsilon.
 * @return false If all elements in the difference matrix are below epsilon.
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
 * @brief Perform mean shift clustering on the given image matrix.
 * 
 * @param hs Spatial threshold.
 * @param hc Chromatic threshold.
 * @param eps Convergence tolerance (not used in this implementation).
 * @param itMax Maximum number of iterations.
 * @param matIMG Image matrix to be processed.
 * @return Mat Segmented image matrix after mean shift clustering.
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
* @brief Detects modes (local maxima) in the segmented image.
 * 
 * This function scans each pixel of the segmented image to detect modes, which are values of color that appear most frequently or are close to values already detected. 
 * Modes are stored in a vector and sorted based on their frequency of appearance.
 * 
 * @param matIMG Matrix of the segmented image after applying the mean shift algorithm.
 * @param eps Tolerance for convergence, indicating the maximum distance to consider two pixels as the same mode.
 * @return vector<Vec3f> List of detected modes, sorted by descending frequency.
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
            float minDist = std::numeric_limits<float>::max();
            Vec3f closestMode;

            // Cherche si ce pixel est proche d'un mode existant
            #pragma omp critical
            {
                for (const auto& mode : modes)
                {
                    float dist = cv::norm(mode - pixel);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        closestMode = mode;
                    }
                }

                // Si un mode proche est trouvé dans la tolérance, l'associer
                if (minDist < eps)
                {
                    modeMap[closestMode]++;
                    found = true;
                }
                else
                {
                    // Sinon, ajouter ce pixel comme un nouveau mode
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
 * @brief Assigns labels to pixels based on detected modes.
 * 
 * This function traverses each pixel of the segmented image and assigns it the label of the nearest mode. The result is an image where each pixel is replaced by the color of the closest mode.
 * 
 * @param matIMG Matrix of the segmented image after applying the mean shift algorithm.
 * @param modes List of detected modes, where each mode represents a color.
 * @param eps Tolerance for convergence (not used in this function).
 * @return Mat Matrix containing the pixel labels, where each pixel is replaced by the color of the closest mode.
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
 * @brief Calculates the Euclidean distance between two 5-channel vectors.
 * 
 * @param a First 5-channel vector.
 * @param b Second 5-channel vector.
 * @return The Euclidean distance between `a` and `b`.
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
 * @brief Calculates the index of the closest center from a 5-channel vector.
 * 
 * @param sample 5-channel vector for which to find the closest center.
 * @param centers List of all centers.
 * @return Index of the closest center in `centers`.
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
 * @brief Executes the k-means algorithm for 5-channel vectors.
 * 
 * This function partitions the input vectors into `k` clusters by minimizing the intra-cluster distance. It returns the centers of the clusters.
 * 
 * @param samples Vector containing the 5-channel vectors to be clustered.
 * @param k Number of clusters.
 * @return vector<Vec<float, 5>> Centers of the `k` clusters.
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
 * @brief Calculates superpixels of the image using the k-means algorithm.
 * 
 * This function converts the image to CIE Lab color space, then runs k-means to segment the image into `k` superpixels. Pixel labels are computed, and the centers of the superpixels are returned.
 * 
 * @param matIMG Input image in BGR format.
 * @param k Number of superpixels to generate.
 * @param m Scaling factor for pixel positions.
 * @param itMax Maximum number of iterations for k-means.
 * @param labels Output matrix containing the pixel labels.
 * @param centers Output matrix containing the centers of the superpixels.
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


    // Filtrer les valeurs NaN et infinies dans centers
    Mat colors;

    // Convertir colors pour K-means
    result.convertTo(colors, CV_32F);
    // reshape the image into a 1D array, for kmeans
    Mat color_reshaped = colors.reshape(3, 1);
    

    // Appliquer K-means pour la binarisation
    Mat labelsKMeans;
    PRINT_MAT_INFO(centers);
    PRINT_MAT_INFO(color_reshaped);
    Mat centersKMeans=kmeans(result, color_reshaped, 2);
    Mat resultbinary(result.size(), CV_8UC3);
        // Loop for each pixel
        for (int i = 0; i < resultbinary.rows; i++)
        {
            for (int j = 0; j < resultbinary.cols; j++)
            {
                Vec3b color = result.at<Vec3b>(i, j);
                int closest = closestCenter(Vec3f(color[0], color[1], color[2]), centersKMeans);
                resultbinary.at<Vec3b>(i, j) = Vec3b(centersKMeans.at<Vec3f>(closest)[0],
                                                     centersKMeans.at<Vec3f>(closest)[1],
                                                     centersKMeans.at<Vec3f>(closest)[2]);
            }
        }
    
    stringstream ss1, ss2;
    ss1 << "superpixel_" << k << ".png";
    ss2 << "binary_superpixel" << k << ".png";
    string outputFile1 = ss1.str();
    string outputFile2 = ss2.str();
    imwrite(outputFile1, result);
    imwrite(outputFile2, resultbinary);

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

            int closest = closestCenter(Vec3f(color[0], color[1], color[2]), centers);

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
                resultMatrix.at<Vec3b>(i, j) = Vec3b(centers.at<Vec3f>(closest)[0],
                                                 centers.at<Vec3f>(closest)[1],
                                                 centers.at<Vec3f>(closest)[2]);
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

    //Mat msMatrix = meanshift(hs, hc, eps, ite, sourceMatrix32f);
    // Détection des modes
    //texture3 et 11 bonnes avec 80
    //tex8 => 35
    //vector<Vec3f> modes = detectModes(msMatrix, 80);

    // Attribution des labels
    //Mat label = labelPixels(msMatrix, modes);
    //Mat msMatrixU;
    //label.convertTo(msMatrixU, CV_8U);

    // Nom du fichier de sortie
    //stringstream ss;
    //ss << "kmeans_" << k << "_classes.png";
    //string outputFile = ss.str();

    // Enregistrer l'image binaire résultante
    //imwrite(outputFile, resultMatrix);

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
    //imshow("Mean Shift", msMatrixU);



    // press q to end
    //while (waitKey(0) != 113);

    // compute superpixels for the image 
    //waitKey(0);
    // Initialize variables for superpixel function
    Mat labels_sup, centers_sup;
    // Paramètres
    int m = 10;
    int k_sup=200;

    // Call the superpixel function
    superpixel(sourceMatrix_sup, k_sup, m, 50, labels_sup, centers_sup);



    return EXIT_SUCCESS;
}
