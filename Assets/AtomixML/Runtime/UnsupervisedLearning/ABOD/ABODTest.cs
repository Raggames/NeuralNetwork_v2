using Atom.MachineLearning.Core;
using Atom.MachineLearning.Core.Maths;
using Atom.MachineLearning.IO;
using Atomix.ChartBuilder;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Unsupervised.AngleBasedOutlierDetection
{
    internal class ABODTest : MonoBehaviour
    {
        [SerializeField] private ABODModel _aBODModel;

        [SerializeField] private VisualizationSheet _visualizationSheet;


        [Button]
        private async void GenerateRandomScatterGraph(int pointsCount = 15, int outliersCount = 2)
        {
            var points = new List<NVector>();
            var maxDist = 10f;
            var outlierMaxDist = 75;

            for (int i = 0; i < pointsCount; i++)
            {
                var distToCenterX = MLRandom.Shared.Range(-maxDist, maxDist);
                var distToCenterY = MLRandom.Shared.Range(-maxDist, maxDist);

                var pos = new NVector(distToCenterX, distToCenterY);
                points.Add(pos);
            }

            var outlierOffset = new NVector(7, 7);

            for (int i = 0; i < outliersCount; i++)
            {
                var distToCenterX = MLRandom.Shared.Range(-outlierMaxDist, outlierMaxDist);
                var distToCenterY = MLRandom.Shared.Range(-outlierMaxDist, outlierMaxDist);

                var pos = outlierOffset + new NVector(distToCenterX, distToCenterY);
                points.Add(pos);
            }


            // shuffle
            var array = points.ToArray();
            DatasetRWUtils.ShuffleRows(array);

            await _aBODModel.Fit(array);

            var dict = new Dictionary<Color, List<NVector>>();
            dict.Add(Color.green, new List<NVector>());
            dict.Add(Color.red, new List<NVector>());

            var results = _aBODModel.GetClassedDatas();
            var cleaned_data = new List<NVector>(); 

            foreach(var data in results)
            {
                if (data.Value)
                {
                    dict[Color.red].Add(data.Key);
                }
                else
                {
                    cleaned_data.Add(data.Key);
                    dict[Color.green].Add(data.Key);
                }
            }

            var input_dict = new Dictionary<Color, double[,]>();
            foreach (var data in dict)
            {
                input_dict.Add(data.Key, data.Value.ToDoubleMatrix());
            }

            _visualizationSheet.Awake();

            // un cleaned
            var root = _visualizationSheet.AddPixelSizedContainer("c0", new Vector2Int(750, 750));
            var container = _visualizationSheet.AddContainer("c0", Color.black, new Vector2Int(100, 100), root);
            container.SetPadding(10, 10, 10, 10);

            //var scatter = _visualizationSheet.Add_Scatter(array.ToDoubleMatrix(), new Vector2Int(100, 100), container);
            var scatter = _visualizationSheet.Add_Scatter(input_dict, new Vector2Int(100, 100), container);

            scatter.gridSize = new Vector2Double(5, 5);
            scatter.gridSizeMode = Atomix.ChartBuilder.VisualElements.ChartBaseElement.GridModes.FixedDeltaValue;

            scatter.SetPadding(50, 50, 50, 50);
            scatter.SetTitle("Random points with outliers");
            scatter.DrawAutomaticGrid();

            // cleaned
 
            var root_cleaned = _visualizationSheet.AddPixelSizedContainer("c0_cleaned", new Vector2Int(750, 750));
            var container_cleaned = _visualizationSheet.AddContainer("c0_cleaned", Color.black, new Vector2Int(100, 100), root_cleaned);
            container_cleaned.SetPadding(10, 10, 10, 10);

            //var scatter = _visualizationSheet.Add_Scatter(array.ToDoubleMatrix(), new Vector2Int(100, 100), container);
            var scatter_cleaned = _visualizationSheet.Add_Scatter(cleaned_data.ToDoubleMatrix(), new Vector2Int(100, 100), container_cleaned);

            scatter_cleaned.gridSize = new Vector2Double(5,5);
            scatter_cleaned.gridSizeMode = Atomix.ChartBuilder.VisualElements.ChartBaseElement.GridModes.FixedDeltaValue;

            scatter_cleaned.SetPadding(50, 50, 50, 50);
            scatter_cleaned.SetTitle("Random points after cleansing");
            scatter_cleaned.DrawAutomaticGrid();
        }
    }
}
