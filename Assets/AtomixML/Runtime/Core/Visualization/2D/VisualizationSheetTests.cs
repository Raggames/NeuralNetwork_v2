using Atom.MachineLearning.Core.Maths;
using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UIElements;

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

            var dimension = new Vector2Int(300, 300);

            // on crée une boite conteneur
            var parent = _visualizationSheet.AddContainer("c1", Color.green, dimension);
            parent.SetPadding(10, 10, 10, 10);
            // on ajoute un graphe qui contient seulement les axis
            var axis = _visualizationSheet.AddAxis("a1", new Color(0, 0, 0, 0), new Vector2Int(100, 100), parent);

            var lineGraph = _visualizationSheet.Add_SimpleLine(points, 3, new Vector2Int(100, 100), axis);
            // permet d s'afficher par dessus les AXIS
            //lineGraph.backgroundColor = new Color(1, 1, 0, 1);
            lineGraph.SetPadding(10, 10, 10, 10);
            lineGraph.Refresh();
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

            var parent = _visualizationSheet.AddContainer("c1", Color.green, new Vector2Int(500, 300));
            parent.SetPadding(10, 10, 10, 10);

            var line = _visualizationSheet.Add_SimpleLine(points, 2, new Vector2Int(500, 300), parent);
            line.style.position = Position.Relative;
            line.style.top = 0;
            line.style.left = 0;
            line.style.right = 0;
            line.style.bottom = 0;

            line.DrawBottomLeftGraduation();
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

            _visualizationSheet.Add_SimpleLine(points, 2, new Vector2Int(300, 300));
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
