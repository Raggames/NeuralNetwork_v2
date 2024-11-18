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
        private async void GenerateDataset(int ratingsCount = 500, int itemCount = 25, double sparsity = .05f, double outlier_ratio = .1f)
        {
            var itDistribCsv = DatasetRWUtils.ReadCSV(_itemTypesDistributionCsvPath, ',', 1);
            _itemTypesDistributions= new FeaturesParser().Transform(new FeaturesSelector(new int[] { 2 }).Transform(itDistribCsv)).Column(0);
          
            var utDistribCsv = DatasetRWUtils.ReadCSV(__userTypesDistributionsCsvPath, ',', 1);
            _userTypesDistributions = new FeaturesParser().Transform(new FeaturesSelector(new int[] { 2 }).Transform(utDistribCsv)).Column(0);

            var uiProfilesCsv = DatasetRWUtils.ReadCSV(_userToItemProfilesCsvPath, ',', 1);
            _userToItemProfiles = new FeaturesParser().Transform(new FeaturesSelector(Enumerable.Range(1, 26).ToArray()).Transform(uiProfilesCsv));


            var dict = new Dictionary<int, string>();
            var parsed = new FeaturesSelector(new int[] { 0, 1 }).Transform(itDistribCsv);
            foreach(var feature in parsed)
            {
                dict.Add(int.Parse(feature.Data[0]), feature.Data[1]);
            }

            var dictUsertypes = new Dictionary<int, string>();  
            var parsed2 = new FeaturesSelector(new int[] { 0, 1 }).Transform(utDistribCsv);
            foreach (var feature in parsed2)
            {
                dict.Add(int.Parse(feature.Data[0]), feature.Data[1]);
            }

            // generate the collection of items of the 'shop' based on a weighted distribution for each type
            var itemType = new int[itemCount];
            var itemTypeString = new string[itemCount];
            var itemNames =new string[itemCount];
            for (int i = 0; i < itemCount; i++)
            {
                itemType[i] = MLRandom.Shared.WeightedIndex(_itemTypesDistributions.Data);
                itemTypeString[i] = itemType[i].ToString();
                itemNames[i] = dict[itemType[i]];
            }

            Debug.Log("Item Types : " + string.Join(", ", itemNames));

            // threshold for the random to have a rating, the higher spartsity, the more 'holes' in the matrix
            double generate_threshold = sparsity;

            // the more outlier_ratio, the more ratings out of profile range 
            double outlier_threshold = 1.0 - outlier_ratio;

            // keep track of user type of each rating to further run checks 
            var user_types = new int[ratingsCount];

            var ratings = new NVector[ratingsCount];
            for (int u = 0; u < ratingsCount; u++)
            {
                int user_type = MLRandom.Shared.WeightedIndex(_userTypesDistributions.Data);
                // based on the user type, we have a profile 
                // we use the min-max range for each item type for the given profile to generate a rating
                user_types[u] = user_type;

                ratings[u] = new NVector(itemType.Length);

                for (int i = 0; i < itemType.Length; ++i)
                {

                    var item_type_min = _userToItemProfiles[user_type][itemType[i] * 2];
                    var item_type_max = _userToItemProfiles[user_type][itemType[i] * 2 + 1];


                    if (MLRandom.Shared.NextDouble() < generate_threshold)
                    {
                        ratings[u][i] = 0;
                        continue;
                    }

                    var base_rating = MLRandom.Shared.Range(item_type_min, item_type_max);
                    ratings[u][i] = base_rating;

                    if (MLRandom.Shared.NextDouble() < outlier_threshold)
                        continue;

                    // add some noise to the rating to make it more realistic
                    var noise = MLRandom.Shared.Range(-2, 2);
                    ratings[u][i] += noise;
                    ratings[u][i] = Mathf.Clamp((float)ratings[u][i], 0, 5);
                }

                Debug.Log($"Generated rating for user type {user_type} : {ratings[u]} ");
            }

            DatasetRWUtils.WriteCSV($"{_outputDatasetCsvPath}_{ratingsCount}_{itemCount}.csv", ';', itemNames, ratings.ToStringMatrix());

            var model = new ItemBasedRecommenderModel("test", ItemBasedRecommenderModel.SimilarityFunctions.AdjustedCosine);
            var result = await model.Fit(ratings);

            Debug.Log("Prediction completed ratings");

            // now compute missing scores
            var prediction_completed_ratings = new NVector[ratings.Length];
            int goodPrediction = 0;
            int allPrediction = 0;
            var mean_error = 0.0;
            for (int i = 0; i < ratings.Length; i++)
            {
                prediction_completed_ratings[i] = model.Predict(ratings[i]);

                Debug.Log($"User {i} > " + prediction_completed_ratings[i].ToString());

                int user_type = user_types[i];

                for (int k = 0; k < prediction_completed_ratings[i].Length; k++)
                {
                    // if it is a prediction
                    if (ratings[i][k] == 0)
                    {
                        var min = _userToItemProfiles[user_type][itemType[k] * 2];
                        var max = _userToItemProfiles[user_type][itemType[k] * 2 + 1];

                        var mid = (max + min) / 2;

                        if (prediction_completed_ratings[i][k] >= min && prediction_completed_ratings[i][k] <= max)
                        {
                            goodPrediction++;
                        }
                        else
                        {
                            Debug.Log($"Prediction {prediction_completed_ratings[i][k]} > min-max {min}-{max}. Item type {itemNames[itemType[k]]}. User Type {dictUsertypes[user_type]}"); 

                        }
                        mean_error += Math.Abs(prediction_completed_ratings[i][k] - mid);
                        allPrediction++;
                    }
                }
            }

            mean_error /= allPrediction;

            // We count prediction as 'good' if it is between the min-max value of the user profile 
            Debug.Log($"The algorithm generated {goodPrediction} good predictions over {allPrediction} predictions. Mean error is {mean_error}");
        }
    }

}
