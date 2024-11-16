using Atom.MachineLearning.Core.Maths;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.Core.Visualization
{
    public class VisualizationSheetTests : MonoBehaviour
    {
        [SerializeField] private VisualizationSheet _visualizationSheet;

        private void Reset()
        {
            _visualizationSheet = GetComponent<VisualizationSheet>();
        }

        [Button]
        private void Test_SimpleLine()
        {
            _visualizationSheet.Awake();
            var points = new double[100];

            for(int i = 0; i < 100; ++i)
            {
                points[i] = Math.Pow(i, 2);
            }

            _visualizationSheet.Add_SimpleLine(points);
        }

        [Button]
        private void Test_SimpleLine2()
        {
            _visualizationSheet.Awake();
            var points = new double[100];

            for (int i = 1; i < 100; ++i)
            {
                points[i] = 1f / Math.Pow(i, 2);
            }

            _visualizationSheet.Add_SimpleLine(points);
        }

        [Button]
        private void Test_SimpleLine3()
        {
            _visualizationSheet.Awake();
            var points = new double[100];

            for (int i = 1; i < 100; ++i)
            {
                points[i] = i;
            }

            _visualizationSheet.Add_SimpleLine(points);
        }


        [Button]
        private void Test_Scatter(int pCount = 100, int X = 50, int Y = 500)
        {
            _visualizationSheet.Awake();

            var points = new double[pCount, 2];

            for (int i = 0; i < pCount; ++i)
            {
                points[i, 0] = MLRandom.Shared.Range(-X, X);
                points[i, 1] = MLRandom.Shared.Range(-Y, Y);
            }

            _visualizationSheet.Add_Scatter(points);
        }
    }
}
