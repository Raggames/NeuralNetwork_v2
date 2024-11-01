using Atom.MachineLearning.Core;
using Atom.MachineLearning.IO;
using MathNet.Numerics.LinearAlgebra;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using Unity.Plastic.Newtonsoft.Json;
using UnityEngine;

namespace Atom.MachineLearning.Unsupervised.PCA
{
    public class PCATrainer : MonoBehaviour, IMLTrainer<PCAModel, NVector, NVector>
    {
        private double[] _means;
        private double[] _stdDeviations;

        /// <summary>
        /// Valus in 0-1 range will be understood as a total explained variance threshold
        /// Integer values above or equal to 1 will be understood as a given count of dimensions
        /// </summary>
        [SerializeField] private float _componentSelectionThreshold;

        [Button]
        private async void TestMNISTFit(string texturesPath = "mnist", int maximumSetSize = 50)
        {
            var model = new PCAModel();
            var textures = DatasetReader.ReadTextures(texturesPath);

            var vectorized = new NVector[textures.Count];
            for (int i = 0; i < textures.Count; ++i)
            {
                if (i > maximumSetSize)
                    break;

                vectorized[i] = new NVector(VectorizationUtils.Texture2DToArray(textures[i]));
            }

            var result = await Fit(model, vectorized);
        }

        [Button]
        private async void TestFitFlowers(string csvpaath = "Assets/AtomixML/Runtime/UnsupervisedLearning/PCA/Resources/flowers/iris.data.txt", int maximumSetSize = 50)
        {
            var model = new PCAModel();
            var datas = DatasetReader.ReadCSV(csvpaath, ',');

            DatasetReader.SplitLastColumn(datas, out var features, out var labels);

            var vectorized_labels = VectorizationUtils.RuledVectorization(labels, 3, new Dictionary<string, double[]>()
            {
                { "Iris-setosa", new double[] { 0, 0, 1 } },
                { "Iris-versicolor", new double[] { 0, 1, 0 } },
                { "Iris-virginica", new double[] { 1, 0, 0 } },
            });

            var vectorized_features = VectorizationUtils.StringMatrix2DToDoubleMatrix2D(features).ToNVectorArray();
            
            var result = await Fit(model, vectorized_features);
        }

        public struct EigenData
        {
            public EigenData(double eigenValue, double[] eigenVector)
            {
                EigenValue = eigenValue;
                EigenVector = eigenVector;
            }

            public double EigenValue { get; set; }
            public double[] EigenVector { get; set; }
        }

        public async Task<ITrainingResult> Fit(PCAModel model, NVector[] trainingDatas)
        {
            var standardizedDatas = NVector.Standardize(trainingDatas, out _means, out _stdDeviations);
            var covariance_matrix = NVector.CovarianceMatrix(standardizedDatas);

            var matrix = Matrix<double>.Build.DenseOfArray(covariance_matrix);
            var evd = matrix.Evd();  // Eigenvalue decomposition

            var eigenvalues = evd.EigenValues.AsArray();
            var eigenvectors = evd.EigenVectors;

            var eigen_datas = new EigenData[eigenvalues.Length];
            var eigen_sum = 0.0;

            for (int i = 0; i < eigenvalues.Length; ++i)
            {
                eigen_datas[i] = new EigenData(eigenvalues[i].Real, eigenvectors.Column(i).AsArray());
                eigen_sum += eigen_datas[i].EigenValue;
            }

            eigen_datas = eigen_datas.OrderByDescending(t => t.EigenValue).ToArray();

            /* switch (_componentsComputationModes)
             {
            // compute purcentage of explained variance
                 case Threshold:
                     break;
            // select a given number of dimensions descending in components energy
                 case Count:
                     break;
             }*/

            var selected_components = new List<EigenData>();

            if (_componentSelectionThreshold >= 0f && _componentSelectionThreshold < 1f)
            {
                // deciding how much dimensions we need
                var tot_variance = 0.0;
                var desired_variance_threshold = _componentSelectionThreshold * 100f; // a purcentage of the total variance
                var threshold_reached = false;

                for (int i = 0; i < eigen_datas.Length; ++i)
                {
                    var explained_variance = (eigen_datas[i].EigenValue / eigen_sum) * 100f;
                    tot_variance += explained_variance;

                    if (!threshold_reached)
                        selected_components.Add(eigen_datas[i]);

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
                    selected_components.Add(eigen_datas[i]);

            }
            else throw new Exception($"The component selection value can't be superior as the total number of dimensions of the input features");

            Debug.Log($"Selected components count : {selected_components.Count}");
            // projection matrix : this will be what the algorithm has learned
            // each feature will then be multiplied by the matrix 

            return new TrainingResult()
            {
                Accuracy = 0,
            };
        }
    }
}
