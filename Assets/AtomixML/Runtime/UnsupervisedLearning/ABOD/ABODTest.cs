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
            foreach (var data in array)
            {
                var predicted_class = _aBODModel.Predict(data);
                if (predicted_class == 1)
                {
                    dict[Color.red].Add(data);
                }
                else
                {
                    dict[Color.green].Add(data);
                }
            }

            var input_dict = new Dictionary<Color, double[,]>();
            foreach (var data in dict)
            {
                input_dict.Add(data.Key, data.Value.ToDoubleMatrix());
            }

            _visualizationSheet.Awake();

            var root = _visualizationSheet.AddPixelSizedContainer("c0", new Vector2Int(750, 750));

            var container = _visualizationSheet.AddContainer("c0", Color.black, new Vector2Int(100, 100), root);
            container.SetPadding(10, 10, 10, 10);

            //var scatter = _visualizationSheet.Add_Scatter(array.ToDoubleMatrix(), new Vector2Int(100, 100), container);
            var scatter = _visualizationSheet.Add_Scatter(input_dict, new Vector2Int(100, 100), container);


            scatter.gridSize = new Vector2Double(8, 8);
            scatter.gridSizeMode = Atomix.ChartBuilder.VisualElements.ChartBaseElement.GridModes.FixedPointsCount;

            scatter.SetPadding(50, 50, 50, 50);
            scatter.SetTitle("Random points with outliers");
            scatter.DrawAutomaticGrid();
        }
    }
}
