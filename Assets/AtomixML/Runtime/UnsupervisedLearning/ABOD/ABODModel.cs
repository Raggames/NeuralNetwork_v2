﻿using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Training;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Unsupervised.AngleBasedOutlierDetection
{
    [Serializable]
    /*
     https://www.dbs.ifi.lmu.de/~zimek/publications/KDD2008/KDD08-ABOD.pdf
     */
    public class ABODModel : IMLModel<NVector, double>, IMLTrainer<ABODModel, NVector, double>
    {
        /// <summary>
        /// The threshold to determine whereas the tested data is an outlier, or not
        /// </summary>
        [SerializeField, HyperParameter] private float _varianceThreshold = 0f;
        [SerializeField, HyperParameter] private bool _predictVariance = true;

        [Space]
        [SerializeField, LearnedParameter] private NVector[] _x_datas;
        [SerializeField, LearnedParameter] private double[,] _angleMatrix;
        [SerializeField, LearnedParameter] private double _population_mean_variance;
        [SerializeField, LearnedParameter] private double _population_variance;
        [SerializeField, LearnedParameter] private Vector2 _min_max_variance;
        [SerializeField, LearnedParameter] private int _permutations_count = 0;

        public string ModelName { get; set; }
        public string ModelVersion { get; set; }
        public ABODModel trainedModel { get; set; }

        public async Task<ITrainingResult> Fit(NVector[] x_datas)
        {
            return await Task.Run(() => FitSynchronously(x_datas));
        }

        public async Task<double> Score()
        {
            return await Task.Run(() => ScoreSynchronously());
        }

        public Dictionary<NVector, double> GetVariances()
        {
            var dict = new Dictionary<NVector, double>();   

            for(int i = 0; i < _angleMatrix.GetLength(0); i++)
            {
                dict.Add(_x_datas[i], _angleMatrix[i, _permutations_count + 2]);
            }

            return dict;
        }

        /// <summary>
        /// True for outliers, false for non outliers
        /// </summary>
        /// <returns></returns>
        public Dictionary<NVector, bool> GetClassedDatas()
        {
            var dict = new Dictionary<NVector, bool>();

            for (int i = 0; i < _angleMatrix.GetLength(0); i++)
            {
                dict.Add(_x_datas[i], _angleMatrix[i, _permutations_count + 2] < _varianceThreshold ? true : false);
            }

            return dict;
        }

        /// <summary>
        /// Predict the variance for one sample
        /// </summary>
        /// <param name="inputData"></param>
        /// <returns></returns>
        public double Predict(NVector inputData)
        {
            int pair_index = 0;

            // to clarify the algorithm
            // we take a point and note it as origin
            // (while predicting, this will be the actual input)
            NVector origin = inputData;
            var angles = new double[_permutations_count + 1];

            for (int j = 0; j < _x_datas.Length - 1; j++)
            {
                if (_x_datas[j] == origin)
                    continue;

                // create a vector OX from origin to a point x (not equal to origin)
                NVector origin_to_x = _x_datas[j] - origin;

                for (int k = j + 1; k < _x_datas.Length; k++)
                {
                    if (k == j)
                        continue;

                    if (_x_datas[k] == origin)
                        continue;

                    // create a vector OY from origin to a point y (x != origin, x != y, y != origin)
                    NVector origin_to_y = _x_datas[k] - origin;

                    // compute OX^OY angle
                    var angle = origin_to_x.CosineAngle(origin_to_y);
                    angles[pair_index] = angle;

                    // last cell of each row is the angle sum
                    // we gain time by computing it along the way
                    angles[_permutations_count] += angle;

                    pair_index++;
                }
            }

            var mean = angles[_permutations_count] / _x_datas.Length;
            var sum = 0.0;
            for (int j = 0; j < _permutations_count - 1; j++)
            {
                sum += Math.Pow(angles[j] - mean, 2);
            }
            var variance = sum / (_permutations_count - 1);

            if (_predictVariance)
                return variance;

            // should return somehow, the ratio 01 of chances that the item is an outlier
            // now its binary, but could we imagine something more continuous ?
            return variance < _varianceThreshold ? 1 : 0;
        }

        public ITrainingResult FitSynchronously(NVector[] x_datas)
        {
            _min_max_variance = new Vector2(float.MaxValue, float.MinValue);
            _x_datas = x_datas;

            _permutations_count = GetRowPermutationsCount(x_datas.Length);

            // we track angle between each permutation of point
            // a row of this matrix represent all angle between the point and every other possible angle permutation between this point and two others
            // we will sum the row to compute the total angle of a point
            _angleMatrix = new double[x_datas.Length, _permutations_count + 3];

            // iterate over the dataset to compute the angle between a point and two other points
            // we only compute angle between two unique pairs of vector            
            for (int i = 0; i < x_datas.Length; i++)
            {
                int pair_index = 0;

                // to clarify the algorithm
                // we take a point and note it as origin
                // (while predicting, this will be the actual input)
                NVector origin = x_datas[i];

                for (int j = 0; j < x_datas.Length - 1; j++)
                {
                    if (j == i)
                        continue;

                    // create a vector OX from origin to a point x (not equal to origin)
                    NVector origin_to_x = x_datas[j] - origin;

                    for (int k = j + 1; k < x_datas.Length; k++)
                    {
                        if (i == j || i == k || k == j)
                            continue;

                        // create a vector OY from origin to a point y (x != origin, x != y, y != origin)
                        NVector origin_to_y = x_datas[k] - origin;

                        // compute OX^OY angle
                        var angle = origin_to_x.CosineAngle(origin_to_y);
                        _angleMatrix[i, pair_index] = angle;

                        // last cell of each row is the angle sum
                        // we gain time by computing it along the way
                        _angleMatrix[i, _permutations_count] += angle;

                        pair_index++;
                    }
                }

                var mean = _angleMatrix[i, _permutations_count] / _x_datas.Length;
                _angleMatrix[i, _permutations_count + 1] = mean; // memorize mean

                var var_sum = 0.0;
                for (int j = 0; j < _permutations_count - 1; j++)
                {
                    var_sum += Math.Pow(_angleMatrix[i, j] - mean, 2);
                }

                var variance = var_sum / (_permutations_count - 1); // sample variance
                _angleMatrix[i, _permutations_count + 2] = variance; // memorize sample variance

                _min_max_variance.x = Math.Min(_min_max_variance.x, (float)variance);
                _min_max_variance.y = Math.Max(_min_max_variance.y, (float)variance);
            }

            // compute mean of variance
            double sum = 0.0;
            for (int i = 0; i < _angleMatrix.GetLength(0); ++i)
            {
                // last cell is the total angle of a point (sum of angle to every other unique vector like AB-AC = BA-CA)
                sum += _angleMatrix[i, _permutations_count + 2];
            }
            _population_mean_variance = sum / _angleMatrix.GetLength(0);

            double sqrd_sum = 0.0;
            for (int i = 0; i < _angleMatrix.GetLength(0); ++i)
            {
                sqrd_sum += Math.Pow(_angleMatrix[i, _permutations_count + 2] - _population_mean_variance, 2);
            }

            _population_variance = sqrd_sum / _angleMatrix.GetLength(0);

            /*
             Calculons la variance de l’ensemble suivant : 2, 7, 3, 12, 9.
            La première étape est de calculer la moyenne. La somme est de 33 et il y a 5 nombres. La moyenne est donc de 33 ÷ 5 =6,6. 
            Il faut ensuite calculer l’écart élevé au carré entre chaque valeur et la moyenne. Par exemple pour la première valeur :
            (2 - 6,6)2 = 21,16
            Les écarts carrés de chaque valeur sont ensuite additionnés :
            21,16 + 0,16 + 12,96 + 29,16 + 5,76 = 69,20
            Cette somme est ensuite divisée par le nombre de valeurs, soit
            69,20 ÷ 5 = 13,84
            La variance est donc de 13,84. Il suffit de trouver la racine carrée pour obtenir l’écart-type : 3,72.
             */

            return new TrainingResult();
        }

        public double ScoreSynchronously()
        {
            return 0;
        }

        [Button]
        public void TestAngleAlg()
        {
            string[] x_datas = new string[] { "A", "B", "C", "D", "E" };
            int row_permutations_counter = GetRowPermutationsCount(x_datas.Length);

            int total_permutations = row_permutations_counter * x_datas.Length;

            Debug.Log(row_permutations_counter);
            Debug.Log(total_permutations);

            // iterate over the dataset to compute the angle between a point and two other points
            // we only compute angle between two unique pairs of vector            
            for (int i = 0; i < x_datas.Length; i++)
            {
                int unique_index = 0;

                for (int j = 0; j < x_datas.Length - 1; j++)
                {
                    if (j == i)
                        continue;

                    // A to B > B - A
                    string current_origin = $"{x_datas[i]}-{x_datas[j]}";

                    for (int k = j + 1; k < x_datas.Length; k++)
                    {
                        if (i == j)
                            continue;

                        if (i == k)
                            continue;

                        if (k == j)
                            continue;

                        // A to C > C - A
                        string current_target = $"{x_datas[i]}-{x_datas[k]}";

                        // compute AB^AC angle
                        Debug.Log($"{current_origin} to {current_target} ({unique_index})");

                        unique_index++;
                    }
                }
            }
        }

        /// <summary>
        /// Get the number of unique permutations for a row.
        /// The angle/variance matrix will be Data.Lenght * RowPermutationCount
        /// </summary>
        /// <param name="rowLenght"></param>
        /// <returns></returns>
        private static int GetRowPermutationsCount(int rowLenght)
        {
            int row_permutations_counter = 0;
            int index = 0;

            while (rowLenght - (2 + index) > 0)
            {
                row_permutations_counter += rowLenght - (2 + index);
                index++;
            }

            return row_permutations_counter;
        }

    }
}
