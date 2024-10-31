using Atom.MachineLearning.Core;
using Atom.MachineLearning.IO;
using MathNet.Numerics.LinearAlgebra;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Unsupervised.PCA
{
    public class PCATrainer : MonoBehaviour, IMLTrainer<PCAModel, NVector, NVector>
    {
        private double[] _means;
        private double[] _stdDeviations;

        [Button]
        private async void TestFit(string texturesPath)
        {
            var model = new PCAModel();
            var textures = DatasetReader.ReadTextures(texturesPath);

            var vectorized = new List<NVector>();
            for(int i = 0; i < textures.Count; ++i)
            {
                vectorized.Add(new NVector(VectorizationUtils.Texture2DToArray(textures[i])));
            }

            var result = await Fit(model, vectorized);   
        }

        public async Task<ITrainingResult> Fit(PCAModel model, List<NVector> trainingDatas)
        {
            var standardizedDatas = Standardize(trainingDatas.ToArray());
            var covariance_matrix = ComputeCovarianceMatrix(standardizedDatas);

            var matrix = Matrix<double>.Build.DenseOfArray(covariance_matrix);
            var evd = matrix.Evd();  // Eigenvalue decomposition

            var eigenvalues = evd.EigenValues;
            var eigenvectors = evd.EigenVectors;



            return new TrainingResult()
            {
                Accuracy = 0,
            };
        }

        private NVector[] Standardize(NVector[] vectors)
        {
            int dimensions = vectors[0].Length;

            _means = new double[dimensions];
            _stdDeviations = new double[dimensions];

            // compute mean for each feature of the n-dimensional vector array
            for (int i = 0; i < dimensions; ++i)
            {
                _means[i] = NVector.FeatureMean(vectors, i);
            }

            // compute standardDeviation for each feature of the n-dimensional vector array
            for (int i = 0; i < dimensions; ++i)
            {
                _stdDeviations[i] = NVector.FeatureStandardDeviation(vectors, _means[i], i);
            }

            // apply standardisation to ech n-vector
            NVector[] result = new NVector[vectors.Length];
            for (int i = 0; i < vectors.Length; ++i)
            {
                result[i] = Standardize(vectors[i]);
            }

            return result;
        }

        private NVector Standardize(NVector vector)
        {
            var result = new NVector(vector.Length);

            for (int j = 0; j < vector.Length; ++j)
            {
                result.Data[j] = (vector[j] - _means[j]) / _stdDeviations[j];
            }

            return result;
        }

        private double[,] ComputeCovarianceMatrix(NVector[] datas)
        {
            int dimensions = datas[0].Length;
            var matrix = new double[datas[0].Length, datas[0].Length];

            // Iterate over each pair of features
            for (int i = 0; i < dimensions; ++i)
            {
                for (int j = 0; j < dimensions; ++j)
                {
                    // Collect all values for features i and j across all vectors
                    double[] featureIValues = new double[datas.Length];
                    double[] featureJValues = new double[datas.Length];

                    for (int k = 0; k < datas.Length; k++)
                    {
                        featureIValues[k] = datas[k][i];
                        featureJValues[k] = datas[k][j];
                    }

                    // Compute covariance between features i and j
                    matrix[i, j] = Covariance(featureIValues, featureJValues);
                }
            }

            return matrix;
        }

        public static double Covariance(double[] featureA, double[] featureB)
        {
            if (featureA.Length != featureB.Length)
                throw new ArgumentException("Feature arrays must have the same length.");

            double meanA = featureA.Average();
            double meanB = featureB.Average();

            double sum = 0.0;

            for (int i = 0; i < featureA.Length; i++)
            {
                sum += (featureA[i] - meanA) * (featureB[i] - meanB);
            }

            return sum / (featureA.Length - 1);  // Using n-1 for sample covariance
        }


    }
}
