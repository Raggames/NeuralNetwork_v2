using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Training;
using Atom.MachineLearning.IO;
using MathNet.Numerics.LinearAlgebra;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;

namespace Atom.MachineLearning.Unsupervised.PCA
{

    public class PCATrainer : MonoBehaviour, IMLTrainer<PCAModel, NVector, NVector>
    {
        /// <summary>
        /// Valus in 0-1 range will be understood as a total explained variance threshold
        /// Integer values above or equal to 1 will be understood as a given count of dimensions
        /// </summary>
        [SerializeField] private float _componentSelectionThreshold;

        public NVector NVector;
        public NMatrix NMatrix;

        private NVector _meanVector;
        private NVector _stdDeviationVector;
        private NVector[] _test_results;
        private Color[] _labelColors;

        [Header("Debug show")]
        [SerializeField] private float _scale = 3f;
        [SerializeField] private RawImage _rawImage;

        public PCAModel trainedModel { get; set; }

        [Button]
        private async void TestMNISTFit(string texturesPath = "Datasets/mnist")
        {
            trainedModel = new PCAModel();
            trainedModel.ModelName = "pca_mnist";
            var textures = DatasetReader.ReadTextures(texturesPath);

            var vectorized = new NVector[textures.Count];
            for (int i = 0; i < textures.Count; ++i)
            {
                var matrix = TransformationUtils.Texture2DToMatrix(textures[i]);
                matrix = TransformationUtils.PoolAverage(matrix, 4, 2);
                var array = TransformationUtils.MatrixToArray(matrix);
                vectorized[i] = new NVector(array);
            }

            var result = await Fit(vectorized);

            ModelSerializer.SaveModel(trainedModel);
            Debug.Log($"End fitting, accuracy (kept variance) => {result.Accuracy}");
        }

        [Button]
        private void TestLoadMNIST()
        {
            trainedModel = ModelSerializer.LoadModel<PCAModel>("pca_mnist");
        }

        [Button]
        private Texture2D TestMNISTCompressionOutput(Texture2D text)
        {
            // this code is all about data transformation/préparation
            // we first get a numerical matrix representation of the image
            var matrix = TransformationUtils.Texture2DToMatrix(text);
            // we pool the matrice to reduce the dimensions of the image from 32x32 to 8x8
            matrix = TransformationUtils.PoolAverage(matrix, 4, 2);
            // we flatten the data to a row vector
            var array = TransformationUtils.MatrixToArray(matrix);
            // instantiating the vector as a struct
            var inputVector = new NVector(array);

            // executing a 'forward' path. dimensions are reduced
            var output_vector = trainedModel.Predict(inputVector);

            // executing a 'backward' path, a very interesting thing that PCAs can do 
            output_vector = trainedModel.Decompress(output_vector);
            var output_to_matrix = TransformationUtils.ArrayToMatrix(output_vector.Data);
            var texture = TransformationUtils.MatrixToTexture2D(output_to_matrix);

            _rawImage.texture = texture;

            return texture;
        }

        [Button]
        private async void TestFitFlowers()
        {
            trainedModel = new PCAModel();
            trainedModel.ModelName = "pca_flowers";

            var datas = Datasets.Flowers_All();

            // splitting labels from x datas
            DatasetReader.SplitLastColumn(datas, out var features, out var labels);

            // transform label column as a vector matrix of nx3 
            // we could also generate a nx1 with class label -1, 0, 1 or anything else, 
            // but that was a practical way to generate colors depending on the class
            var vectorized_labels = TransformationUtils.Encode(labels, 3, new Dictionary<string, double[]>()
            {
                { "Iris-setosa", new double[] { 0, 0, 1 } },
                { "Iris-versicolor", new double[] { 0, 1, 0 } },
                { "Iris-virginica", new double[] { 1, 0, 0 } },
            });

            _labelColors = new Color[vectorized_labels.GetLength(0)];

            for (int i = 0; i < vectorized_labels.GetLength(0); ++i)
                _labelColors[i] = new Color((float)vectorized_labels[i, 0], (float)vectorized_labels[i, 1], (float)vectorized_labels[i, 2], 1);

            var vectorized_features = TransformationUtils.StringMatrix2DToDoubleMatrix2D(features).ToNVectorRowsArray();

            var result = await Fit(vectorized_features);

            Debug.Log($"End fitting, accuracy (kept variance) => {result.Accuracy}");

            _test_results = new NVector[vectorized_features.Length];
            for (int i = 0; i < vectorized_features.Length; ++i)
            {
                _test_results[i] = trainedModel.Predict(vectorized_features[i]);
            }

            ModelSerializer.SaveModel(trainedModel);
        }

        public async Task<ITrainingResult> Fit(NVector[] trainingDatas)
        {
            var standardizedDatas = NVector.Standardize(trainingDatas, out _meanVector, out _stdDeviationVector, out _);
            var covariance_matrix = NVector.CovarianceMatrix(standardizedDatas);

            var matrix = Matrix<double>.Build.DenseOfArray(covariance_matrix);
            var evd = matrix.Evd();  // Eigenvalue decomposition

            var eigenvalues = evd.EigenValues.AsArray();
            var eigenvectors = evd.EigenVectors;

            var eigen_datas = new EigenPair[eigenvalues.Length];
            var eigen_sum = 0.0;

            for (int i = 0; i < eigenvalues.Length; ++i)
            {
                eigen_datas[i] = new EigenPair(eigenvalues[i].Real, eigenvectors.Column(i).AsArray());
                eigen_sum += eigen_datas[i].EigenValue;
            }

            eigen_datas = eigen_datas.OrderByDescending(t => t.EigenValue).ToArray();

            var selected_components = new List<EigenPair>();

            var kept_variance = 0.0;
            var tot_variance = 0.0;

            if (_componentSelectionThreshold >= 0f && _componentSelectionThreshold < 1f)
            {
                // deciding how much dimensions we need
                var desired_variance_threshold = _componentSelectionThreshold * 100f; // a purcentage of the total variance
                var threshold_reached = false;

                for (int i = 0; i < eigen_datas.Length; ++i)
                {
                    var explained_variance = (eigen_datas[i].EigenValue / eigen_sum) * 100f;
                    tot_variance += explained_variance;

                    if (!threshold_reached)
                    {
                        selected_components.Add(eigen_datas[i]);
                        kept_variance = tot_variance;
                    }

                    // We add component until we reach the minimal variance threshold we want
                    // the count of components will be the dimensions of our projection matrix
                    if (tot_variance > desired_variance_threshold)
                    {
                        threshold_reached = true;
                    }

                    Debug.Log($"Eigen value : {eigen_datas[i].EigenValue}. Explained variance : {explained_variance} / {tot_variance} %");
                }
            }
            else if (_componentSelectionThreshold < eigen_datas.Length)
            {
                int comp = (int)Math.Round(_componentSelectionThreshold);
                for (int i = 0; i < comp; ++i)
                {
                    var explained_variance = (eigen_datas[i].EigenValue / eigen_sum) * 100f;
                    tot_variance += explained_variance;
                    kept_variance = tot_variance;
                    selected_components.Add(eigen_datas[i]);
                }
            }
            else throw new Exception($"The component selection value can't be superior as the total number of dimensions of the input features");

            Debug.Log($"Selected components count : {selected_components.Count}");
            // projection matrix : this will be what the algorithm has learned
            // each feature will then be multiplied by the matrix 

            var projectionMatrix = NMatrix.DenseOfColumnVectors(selected_components.Select(t => t.EigenVector).ToArray());

            trainedModel.Initialize(projectionMatrix, _meanVector, _stdDeviationVector);

            return new TrainingResult()
            {
                Accuracy = (float)kept_variance,
            };
        }

        public Task<double> Score(NVector[] x_datas)
        {
            throw new NotImplementedException();
        }

        void OnDrawGizmos()
        {
            if (_test_results == null)
                return;

            for (int i = 0; i < _test_results.Length; ++i)
            {
                Gizmos.color = _labelColors[i];
                Gizmos.DrawSphere(new UnityEngine.Vector3((float)_test_results[i].Data[0] * _scale, (float)_test_results[i].Data[1], 0) * _scale, .15f);

            }
        }

    }
}
