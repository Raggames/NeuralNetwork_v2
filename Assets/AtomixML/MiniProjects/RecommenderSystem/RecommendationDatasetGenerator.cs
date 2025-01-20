using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Transformers;
using Atom.MachineLearning.IO;
using Atom.MachineLearning.Supervised.Recommender.ItemBased;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using static Atom.MachineLearning.Supervised.Recommender.ItemBased.ItemBasedRecommenderModel;

namespace Atom.MachineLearning.MiniProjects.RecommenderSystem
{

    public class RecommendationDatasetGenerator : MonoBehaviour
    {
        [SerializeField] private string _itemTypesDistributionCsvPath = "Assets/AtomixML/MiniProjects/RecommenderSystem/Resources/itDistrib.csv";
        [SerializeField] private string __userTypesDistributionsCsvPath = "Assets/AtomixML/MiniProjects/RecommenderSystem/Resources/utDistrib.csv";
        [SerializeField] private string _userToItemProfilesCsvPath = "Assets/AtomixML/MiniProjects/RecommenderSystem/Resources/uiProf.csv";
        [SerializeField] private string _outputDatasetCsvPath = "Assets/AtomixML/MiniProjects/RecommenderSystem/Resources/dataset";

        [SerializeField] private NVector _itemTypesDistributions;
        [SerializeField] private NVector _userTypesDistributions;
        [SerializeField] private NVector[] _userToItemProfiles;

        [Button]
        private async void GenerateDataset(int ratingsCount = 50, int itemCount = 10, double sparsity = .3f, double outlier_ratio = .1f, double epsilon = .75f, SimilarityFunctions similarityFunction = SimilarityFunctions.AdjustedCosine)
        {
            MLRandom.SeedShared(DateTime.Now.Millisecond * DateTime.Now.Second);

            // sparsity = inverse of density of ratings per user
            // outlier = density of 'out profile range' ratings over dataset
            // espilon = minimum value to consider rating as an 'error' (too far from profile)

            var itDistribCsv = DatasetRWUtils.ReadCSV(_itemTypesDistributionCsvPath, ',', 1);
            _itemTypesDistributions = new FeaturesParser().Transform(new FeaturesSelector(new int[] { 2 }).Transform(itDistribCsv)).Column(0);

            var utDistribCsv = DatasetRWUtils.ReadCSV(__userTypesDistributionsCsvPath, ',', 1);
            _userTypesDistributions = new FeaturesParser().Transform(new FeaturesSelector(new int[] { 2 }).Transform(utDistribCsv)).Column(0);

            var uiProfilesCsv = DatasetRWUtils.ReadCSV(_userToItemProfilesCsvPath, ',', 1);
            _userToItemProfiles = new FeaturesParser().Transform(new FeaturesSelector(Enumerable.Range(1, 26).ToArray()).Remap("0", "0,5").Transform(uiProfilesCsv));


            var dict = new Dictionary<int, string>();
            var parsed = new FeaturesSelector(new int[] { 0, 1 }).Transform(itDistribCsv);
            foreach (var feature in parsed)
            {
                dict.Add(int.Parse(feature.Data[0]), feature.Data[1]);
            }

            var dictUsertypes = new Dictionary<int, string>();
            var parsed2 = new FeaturesSelector(new int[] { 0, 1 }).Transform(utDistribCsv);
            string[] user_type_names = new string[ratingsCount];

            foreach (var feature in parsed2)
            {
                dictUsertypes.Add(int.Parse(feature.Data[0]), feature.Data[1]);
            }

            // generate the collection of items of the 'shop' based on a weighted distribution for each type
            var item_types = new int[itemCount];
            var itemTypeString = new string[itemCount];
            var itemNames = new string[itemCount];
            for (int i = 0; i < itemCount; i++)
            {
                item_types[i] = MLRandom.Shared.WeightedIndex(_itemTypesDistributions.Data);
                itemTypeString[i] = item_types[i].ToString();
                itemNames[i] = dict[item_types[i]];
            }

            Debug.Log("Item Types : " + string.Join(", ", itemNames));

            // threshold for the random to have a rating, the higher spartsity, the more 'holes' in the matrix
            double generate_threshold = sparsity;

            // the more outlier_ratio, the more ratings out of profile range 
            double outlier_threshold = 1.0 - outlier_ratio;

            // keep track of user type of each rating to further run checks 
            var user_types = new int[ratingsCount];

            var ratings = new NVector[ratingsCount];

            // generating a second matrix that represent the mean value for each user-item if no rating for user (depending on the user profile), and the user actual rating otherwise
            // this set can be later used to check how the encoder is close from the average user (for ranking)
            // this set will be a bunch of duplicated datas, but that's for the sake of simplicity
            var ratings_profile_means = new NVector[ratingsCount];

            for (int u = 0; u < ratingsCount; u++)
            {
                int user_type = MLRandom.Shared.WeightedIndex(_userTypesDistributions.Data);
                // based on the user type, we have a profile 
                // we use the min-max range for each item type for the given profile to generate a rating
                user_types[u] = user_type;
                string user_type_name = dictUsertypes[user_type];

                user_type_names[u] = user_type_name;
              
                ratings[u] = new NVector(item_types.Length);
                ratings_profile_means[u] = new NVector(item_types.Length);

                for (int i = 0; i < item_types.Length; ++i)
                {
                    string item_type_name = dict[item_types[i]];

                    var item_type_min = _userToItemProfiles[user_type][item_types[i] * 2];
                    var item_type_max = _userToItemProfiles[user_type][item_types[i] * 2 + 1];


                    if (MLRandom.Shared.NextDouble() < generate_threshold)
                    {
                        ratings[u][i] = 0;

                        // profile mean if no value for user
                        ratings_profile_means[u][i] = item_type_min + (item_type_max - item_type_min) / 2f;

                        continue;
                    }

                    var base_rating = MLRandom.Shared.Range(item_type_min, item_type_max);
                    ratings[u][i] = base_rating;

                    // actual value if there is one
                    ratings_profile_means[u][i] = ratings[u][i];

                    if (MLRandom.Shared.NextDouble() < outlier_threshold)
                        continue;

                    // add some noise to the rating to make it more realistic
                    var noise = MLRandom.Shared.Range(-2, 2);
                    ratings[u][i] += noise;
                    ratings[u][i] = Mathf.Clamp((float)ratings[u][i], 0, 5);

                    ratings_profile_means[u][i] = ratings[u][i];
                }

                //Debug.Log($"Generated rating for user type {user_type} : {ratings[u]} ");
            }

            DatasetRWUtils.WriteCSV($"{_outputDatasetCsvPath}_{ratingsCount}_{itemCount}.csv", ';', itemNames, user_type_names, ratings.ToStringMatrix());
            DatasetRWUtils.WriteCSV($"{_outputDatasetCsvPath}_{ratingsCount}_{itemCount}_profile_means.csv", ';', itemNames, null, ratings_profile_means.ToStringMatrix());

            var usertypes_matrix = new string[ratings.Length, 1];
            for (int i = 0; i < usertypes_matrix.GetLength(0); i++)
                usertypes_matrix[i, 0] = user_types[i].ToString();

            DatasetRWUtils.WriteCSV($"{_outputDatasetCsvPath}_{ratingsCount}_{itemCount}_check.csv", ';', new string[] { "user_types" }, null, usertypes_matrix);

            var model = new ItemBasedRecommenderModel("test", similarityFunction);
            var result = await model.Fit(ratings);

            ComputePredictionsTest(epsilon, dictUsertypes, dict, item_types, itemNames, user_types, ratings, model);
        }

        private void ComputePredictionsTest(double epsilon, Dictionary<int, string> dictUsertypes, Dictionary<int, string> dictItemtypes, int[] item_types, string[] itemNames, int[] user_types, NVector[] ratings, ItemBasedRecommenderModel model)
        {
            Debug.Log("Prediction-completed ratings");

            // now compute missing scores
            var prediction_completed_ratings = new NVector[ratings.Length];
            int good_prediction_count = 0;
            int all_predictions_count = 0;
            var mean_error = 0.0;
            for (int u = 0; u < ratings.Length; u++)
            {
                prediction_completed_ratings[u] = model.Predict(ratings[u]);

                Debug.Log($"User {u} / {dictUsertypes[user_types[u]]} > " + prediction_completed_ratings[u].ToString());

                int user_type = user_types[u];

                for (int i = 0; i < item_types.Length; ++i)
                {
                    int item_type = item_types[i];

                    // if it is a prediction
                    if (ratings[u][i] == 0)
                    {
                        // since we generate the ratings from user profiles, we know exactly where the predicted value should be 
                        var min_rating = _userToItemProfiles[user_type][item_type * 2];
                        var max_rating = _userToItemProfiles[user_type][item_type * 2 + 1];

                        // we compute a threshold for good/bad prediction
                        var delta = (max_rating - min_rating) / 2 + epsilon;

                        var mean = (max_rating + min_rating) / 2;
                        var crt_absolute_error = Math.Abs(prediction_completed_ratings[u][i] - mean);
                        if (crt_absolute_error <= delta)
                        {
                            good_prediction_count++;

                            Debug.Log($"Prediction " +
                               $"{prediction_completed_ratings[u][i]} > min-max {min_rating}-{max_rating}. " +
                               $"Item type {itemNames[i]}. " +
                               $"User Type {dictUsertypes[user_type]}");
                        }
                        else
                        {
                            Debug.LogError($"Prediction " +
                                $"{prediction_completed_ratings[u][i]} > min-max {min_rating}-{max_rating}. " +
                                $"Item type {itemNames[i]}. " +
                                $"User Type {dictUsertypes[user_type]}");
                        }

                        // summing the absolute value of predicted - mean profile rating
                        // we multiply by rating delta to increase error for higher range, and minimise error for short range
                        mean_error += crt_absolute_error;
                        all_predictions_count++;
                    }
                }
            }

            mean_error /= all_predictions_count;

            // We count prediction as 'good' if it is between the min-max value of the user profile 
            Debug.Log($"The algorithm generated {good_prediction_count} good predictions over {all_predictions_count} predictions. Mean error is {mean_error}");
        }


    }

}
