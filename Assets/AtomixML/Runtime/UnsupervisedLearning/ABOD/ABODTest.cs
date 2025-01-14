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
        private void GenerateRandomScatterGraph(int pointsCount = 100, int outliersCount = 10)
        {
            var points = new List<NVector>();
            var maxDist = 10f;
            var outlierMaxDist = 20f;

            for(int i = 0; i < pointsCount; i++)
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

            _visualizationSheet.Awake();

            var container = _visualizationSheet.AddContainer("c0", Color.black, new Vector2Int(750, 750));
            container.SetPadding(25, 25, 25, 25);
            var scatter = _visualizationSheet.Add_Scatter(array.ToDoubleMatrix(), new Vector2Int(100, 100), container);
            scatter.SetTitle("Random points with outliers");
            scatter.DrawAutomaticGrid();
        }
    }
}
