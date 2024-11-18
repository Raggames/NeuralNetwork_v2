using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Training;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

/*
 https://towardsdatascience.com/comprehensive-guide-on-item-based-recommendation-systems-d67e40e2b75d
 https://medium.com/geekculture/overview-of-item-item-collaborative-filtering-recommendation-system-64ee15b24bb8 

 https://nagendranukala.medium.com/collaborative-filtering-similarity-calculations-a974ae4650
 */

namespace Atom.MachineLearning.Supervised.Recommender.ItemBased
{
    public partial class ItemBasedRecommenderModel : IMLModel<NVector, NVector>, IMLTrainer<ItemBasedRecommenderModel, NVector, NVector>
    {
        public string ModelName { get; set; }
        public string ModelVersion { get; set; }

        public ItemBasedRecommenderModel trainedModel { get; set; }

        // matrice of items (column) and user (rows) > cell is rating for user i on item j
        [SerializeField, LearnedParameter] private NMatrix _userItemRawMatrix;

        // average rating of each item
        [SerializeField, LearnedParameter] private NVector _itemsAverageRatings;

        // similarity matrix computed with Adjusted Cosine similarity function
        [SerializeField, LearnedParameter] private NMatrix _itemSimilarityMatrix;

        // which similarity function is used
        [SerializeField, LearnedParameter] private SimilarityFunctions _similarityFunction;

        public NMatrix itemSimilarityMatrix => _itemSimilarityMatrix;

        public ItemBasedRecommenderModel(string modelName, SimilarityFunctions similarityFunction)
        {
            ModelName = modelName;
            trainedModel = this;
            _similarityFunction = similarityFunction;
        }

        public NVector Predict(NVector userRatings)
        {
            switch (_similarityFunction)
            {
                case SimilarityFunctions.Cosine: return PredictCosine(userRatings);
                case SimilarityFunctions.AdjustedCosine: return PredictAdjustedCosine(userRatings);
            }

            throw new NotImplementedException();
        }

        private NVector PredictCosine(NVector userRatings)
        {
            NVector result = new NVector(userRatings.Length);

            // predict will be done by computing a score foreach item that user hasn't rated
            // the input data is a sparse row vector so we will ignore each feature that is not 0 (0 mean no rating)
            // the predicted score will be an average of rating related to the predicted item, ponderated by the similarity

            var mean = userRatings.Average();
            // mean = 0 means user has no rating at all,
            // we will give him an average of rating of each item above every user (calculated while fitting)
            // it says at least which item is prefered by every other user, so that's a starting point
            if (mean == 0)
            {
                return _itemsAverageRatings;
            }

            for (int i = 0; i < userRatings.Length; i++)
            {
                // we have to predict only missing cases
                // predicting 'known' rating will be used for scoring the model
                if (userRatings[i] == 0.0)
                {
                    // so we are at the ith item here.
                    // we want to know the rating for this user, so we will compare with the rating of similar items

                    double r_s_sum = 0.0;
                    double s_sum = 0.0;

                    for (int j = 0; j < userRatings.Length; j++)
                    {
                        if (i == j) // || userRatings[i] == 0 || userRatings[j] == 0 ? do we take in account other missing rate ? 
                            continue;

                        r_s_sum += userRatings[j] * _itemSimilarityMatrix[i, j];
                        s_sum += _itemSimilarityMatrix[i, j];
                    }

                    result[i] = r_s_sum / s_sum;
                }
                else
                {
                    result[i] = userRatings[i];
                }
            }

            return result;
        }

        private NVector PredictAdjustedCosine(NVector userRatings)
        {
            NVector result = new NVector(userRatings.Length);
            var mean = userRatings.Average();

            // mmean = 0 means user has no rating at all,
            // we will give him an average of rating of each item above every user
            // it says at least which item is prefered by every other user, so that's a starting point
            if (mean == 0)
            {
                return _itemsAverageRatings;
            }

            var sparse_mean = userRatings.SparseAverage();

            // predict will be done by computing a score foreach item that user hasn't rated
            // the input data is a sparse row vector so we will ignore each feature that is not 0 (0 mean no rating)
            // the predicted score will be an average of rating related to the predicted item, ponderated by the similarity
            for (int i = 0; i < userRatings.Length; i++)
            {
                // we have to predict only missing cases
                // predicting 'known' rating will be used for scoring the model
                if (userRatings[i] == 0.0)
                {
                    double weightedSum = 0.0;
                    double similaritySum = 0.0;

                    for (int j = 0; j < userRatings.Length; j++)
                    {
                        if (i != j && userRatings[j] > 0)
                        {
                            var similarity = _itemSimilarityMatrix[i, j];
                            weightedSum += (userRatings[j] - sparse_mean) * similarity;
                            similaritySum += Math.Abs(similarity);
                        }
                    }

                    result[i] = similaritySum != 0 ? (weightedSum / similaritySum) + sparse_mean : sparse_mean;
                }
                else
                {
                    result[i] = userRatings[i];
                }
            }

            return result;
        }

        public async Task<ITrainingResult> Fit(NVector[] x_datas)
        {
            // we keep the training datas that will serve for prediction (which is in fact a simple similarity-ponderated interpolation) 
            _userItemRawMatrix = NMatrix.FromNVectorArray(x_datas);
            _itemsAverageRatings = new NVector(x_datas[0].Length);
            _itemSimilarityMatrix = new NMatrix(x_datas[0].Length, x_datas[0].Length);

            ComputeItemMeanRatings(x_datas);

            // x_datas are a collection of user ratings foreach item
            // we have to compute the similarity matrix (item-item)
            switch (_similarityFunction)
            {
                case SimilarityFunctions.Cosine:
                    ComputeCosineSimilarityMatrix(x_datas);
                    break;
                case SimilarityFunctions.AdjustedCosine:
                    ComputeAdjustedCosineSimilarityMatrix(x_datas);
                    break;
            }

            return new TrainingResult() { Accuracy = 0 };
        }

        private void ComputeItemMeanRatings(NVector[] x_datas)
        {
            for (int i = 0; i < x_datas[0].Length; ++i)
            {
                int count = 0;
                for (int u = 0; u < x_datas.GetLength(0); u++)
                {
                    if (x_datas[u][i] == 0.0) continue;

                    count++;
                    _itemsAverageRatings[i] += x_datas[u][i];
                }

                if (count == 0)
                    _itemsAverageRatings[i] = 0;
                else
                    _itemsAverageRatings[i] /= count;
            }
        }

        private void ComputeCosineSimilarityMatrix(NVector[] x_datas)
        {
            for (int i = 0; i < _itemSimilarityMatrix.Rows; ++i)
            {
                for (int j = 0; j < _itemSimilarityMatrix.Columns; ++j)
                {
                    if (i == j)
                    {
                        _itemSimilarityMatrix[i, j] = 1.0; // we dont have to compute the similarity between the same item, it is 1
                        continue;
                    }

                    var dot_product = 0.0;
                    NVector vec_i = new NVector(x_datas.Length);
                    NVector vec_j = new NVector(x_datas.Length);
                    // adjusetd cosine similarity will be dotproduct ponderated by the user mean rating for each feature of the item vector (column vector of 1 rating per user for each item)
                    for (int k = 0; k < x_datas.Length; ++k)
                    {
                        dot_product += (x_datas[k][i]) * (x_datas[k][j]);

                        // we create the column vector by iterating the row and selecting column index i and j to gain computationnal time
                        vec_i[k] = x_datas[k][i];
                        vec_j[k] = x_datas[k][j];
                    }

                    // product of magnitudes 
                    var norm_produt = vec_i.magnitude * vec_j.magnitude;

                    _itemSimilarityMatrix[i, j] = dot_product / norm_produt;
                }
            }
        }

        private void ComputeAdjustedCosineSimilarityMatrix(NVector[] x_datas)
        {/*
            // as we use adjusted cosine method, we first compute mean rating foreach user
            double[] userMeanRatings = new double[x_datas.Length];
            for (int i = 0; i < x_datas.Length; ++i)
            {
                userMeanRatings[i] = x_datas[i].SparseAverage(); // average rating of the rated items by the user
            }

            _itemSimilarityMatrix = new NMatrix(x_datas[0].Length, x_datas[0].Length);

            for (int i = 0; i < _itemSimilarityMatrix.Rows; ++i)
            {
                for (int j = 0; j < _itemSimilarityMatrix.Columns; ++j)
                {
                    var dot_product = 0.0;
                    NVector vec_i = new NVector(x_datas.Length);
                    NVector vec_j = new NVector(x_datas.Length);

                    // adjusetd cosine similarity will be dotproduct ponderated by the user mean rating for each feature of the item vector (column vector of 1 rating per user for each item)
                    for (int k = 0; k < x_datas.Length; ++k)
                    {
                        if (x_datas[k][i] == 0.0 || x_datas[k][j] == 0.0)
                            continue;

                        dot_product += (x_datas[k][i] - userMeanRatings[k]) * (x_datas[k][j] - userMeanRatings[k]);

                        // we create the column vector by iterating the row and selecting column index i and j to gain computationnal time
                        vec_i[k] = x_datas[k][i];
                        vec_j[k] = x_datas[k][j];
                    }

                    // product of magnitudes 
                    //var norm_product = vec_i.magnitude * vec_j.magnitude;

                    double norm_i = 0.0, norm_j = 0.0;
                    for (int k = 0; k < x_datas.Length; ++k)
                    {
                        if (x_datas[k][i] != 0.0)
                            norm_i += Math.Pow(x_datas[k][i] - userMeanRatings[k], 2);

                        if (x_datas[k][j] != 0.0)
                            norm_j += Math.Pow(x_datas[k][j] - userMeanRatings[k], 2);
                    }
                    var norm_product = Math.Sqrt(norm_i) * Math.Sqrt(norm_j);

                    _itemSimilarityMatrix[i, j] = dot_product / norm_product;
                }
            }*/

            int itemCount = x_datas[0].Length;
            double[] userMeanRatings = x_datas.Select(user => user.SparseAverage()).ToArray();
            _itemSimilarityMatrix = new NMatrix(itemCount, itemCount);

            for (int i = 0; i < itemCount; i++)
            {
                for (int j = 0; j < itemCount; j++)
                {
                    if (i == j)
                    {
                        _itemSimilarityMatrix[i, j] = 1.0;
                        continue;
                    }

                    double dotProduct = 0.0;
                    double normI = 0.0, normJ = 0.0;

                    for (int k = 0; k < x_datas.Length; k++)
                    {
                        var ratingI = x_datas[k][i];
                        var ratingJ = x_datas[k][j];

                        if (ratingI > 0 && ratingJ > 0)
                        {
                            var adjustedI = ratingI - userMeanRatings[k];
                            var adjustedJ = ratingJ - userMeanRatings[k];
                            dotProduct += adjustedI * adjustedJ;
                            normI += Math.Pow(adjustedI, 2);
                            normJ += Math.Pow(adjustedJ, 2);
                        }
                    }

                    var magnitudeProduct = Math.Sqrt(normI) * Math.Sqrt(normJ);
                    _itemSimilarityMatrix[i, j] = magnitudeProduct > 0 ? dotProduct / magnitudeProduct : 0;
                }
            }
        }

        /// <summary>
        /// RMSE or MAE can be used here
        /// </summary>
        /// <param name="x_datas"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public Task<double> Score(NVector[] x_datas)
        {
            throw new NotImplementedException();
        }
    }
}
