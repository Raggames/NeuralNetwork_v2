using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.Core.Transformers;
using Atom.MachineLearning.IO;
using Sirenix.OdinInspector;
using System.Linq;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.RecommenderSystem
{

    public class RecommendationDatasetGenerator : MonoBehaviour
    {
        [SerializeField] private string _itemTypesDistributionCsvPath = "Assets/AtomixML/MiniProjects/RecommenderSystem/Resources/itDistrib.csv";
        [SerializeField] private string __userTypesDistributionsCsvPath = "Assets/AtomixML/MiniProjects/RecommenderSystem/Resources/utDistrib.csv";
        [SerializeField] private string _userToItemProfilesCsvPath = "Assets/AtomixML/MiniProjects/RecommenderSystem/Resources/uiProf.csv";

        [SerializeField] private NVector _itemTypesDistributions;
        [SerializeField] private NVector _userTypesDistributions;
        [SerializeField] private NVector[] _userToItemProfiles;

        [Button]
        private void GenerateDataset(int ratingsCount = 500, int itemCount = 25, double sparsity = .05f, double outlier_ratio = .1f)
        {
            var itDistribCsv = DatasetRWUtils.ReadCSV(_itemTypesDistributionCsvPath, ',', 1);
            _itemTypesDistributions = new FeaturesParser().Transform(new FeaturesSelector(new int[] { 2 }).Transform(itDistribCsv)).Column(0);

            var utDistribCsv = DatasetRWUtils.ReadCSV(__userTypesDistributionsCsvPath, ',', 1);
            _userTypesDistributions = new FeaturesParser().Transform(new FeaturesSelector(new int[] { 2 }).Transform(utDistribCsv)).Column(0);

            var uiProfilesCsv = DatasetRWUtils.ReadCSV(_userToItemProfilesCsvPath, ',', 1);
            _userToItemProfiles = new FeaturesParser().Transform(new FeaturesSelector(Enumerable.Range(1, 26).ToArray()).Transform(uiProfilesCsv));

            // generate the collection of items of the 'shop' based on a weighted distribution for each type
            var itemType = new int[itemCount];
            var itemTypeString = new string[itemCount];
            for (int i = 0; i < itemCount; i++)
            {
                itemType[i] = MLRandom.Shared.WeightedIndex(_itemTypesDistributions.Data);
                itemTypeString[i] = itemType[i].ToString();
            }

            // threshold for the random to have a rating, the higher spartsity, the more 'holes' in the matrix
            double generate_threshold = 1.0 - sparsity;

            // the more outlier_ratio, the more ratings out of profile range 
            double outlier_threshold = 1.0 - outlier_ratio;

            var ratings = new double[ratingsCount, _itemTypesDistributions.Length];
            for (int u = 0; u < ratingsCount; u++)
            {
                int user_type = MLRandom.Shared.WeightedIndex(_userTypesDistributions.Data);
                // based on the user type, we have a profile 
                // we use the min-max range for each item type for the given profile to generate a rating

                for (int i = 0; i < itemType.Length; ++i)
                {
                    var item_type_min = _userToItemProfiles[user_type][itemType[u] * i];
                    var item_type_max = _userToItemProfiles[user_type][itemType[u] * (i + 1)];

                    if (MLRandom.Shared.NextDouble() < generate_threshold)
                    {
                        ratings[u, i] = 0;
                        continue;
                    }


                    var base_rating = MLRandom.Shared.Range(item_type_min, item_type_max);
                    ratings[u, i] = base_rating;

                    if (MLRandom.Shared.NextDouble() < outlier_threshold)
                        continue;

                    // add some noise to the rating to make it more realistic
                    var noise = MLRandom.Shared.Range(-0.2, 0.2);
                    ratings[u, i] += noise;
                }
            }
        }
    }

}
