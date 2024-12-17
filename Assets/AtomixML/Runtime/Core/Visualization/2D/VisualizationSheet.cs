using Atom.MachineLearning.Core.Visualization.VisualElements;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UIElements;

namespace Atom.MachineLearning.Core.Visualization
{
    public class VisualizationSheet : MonoBehaviour
    {
        [SerializeField] private UIDocument _document;
        private VisualElement _root;

        private static VisualizationSettings _visualizationSettings;
        public static VisualizationSettings visualizationSettings
        {
            get
            {
                if (_visualizationSettings == null)
                {
                    _visualizationSettings = Resources.Load<VisualizationSettings>(nameof(VisualizationSettings));
                }

                return _visualizationSettings;
            }
        }

        public void Awake()
        {
            if (_document == null)
                _document = GetComponent<UIDocument>();

            _root = _document.rootVisualElement;
            _root.Clear();
        }

        public SimpleLineChart Add_SimpleLine(double[,] matrice, float lineWidth, Vector2Int dimensions)
        {
            var chart = new SimpleLineChart(matrice, lineWidth, dimensions.x, dimensions.y);

            _root.Add(chart);
            chart.Refresh();

            return chart;
        }

        public SimpleLineChart Add_SimpleLine(double[] points, float lineWidth, Vector2Int dimensions)
        {
            var chart = new SimpleLineChart(points, lineWidth, dimensions.x, dimensions.y);

            _root.Add(chart);
            chart.Refresh();

            return chart;
        }

        public SimpleLineChart Add_SimpleLine(Func<List<Vector2>> getPoints, float lineWidth, Vector2Int dimensions)
        {
            var chart = new SimpleLineChart(getPoints, lineWidth, dimensions.x, dimensions.y);

            _root.Add(chart);
            chart.Refresh();

            return chart;
        }

        public SimpleLineChart Add_SimpleLine(Func<List<double>> getPoints, float lineWidth, Vector2Int dimensions)
        {
            var chart = new SimpleLineChart(getPoints, lineWidth, dimensions.x, dimensions.y);

            _root.Add(chart);
            chart.Refresh();

            return chart;
        }


        public Scatter2DChart Add_Scatter(double[,] matrice)
        {
            var chart = new Scatter2DChart(() => matrice);
            _root.Add(chart);
            chart.Refresh();
            return chart;
        }
    }
}
