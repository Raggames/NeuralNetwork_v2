using Atom.MachineLearning.Core.Visualization.VisualElements;
using Atom.MachineLearning.IO;
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
            _document.panelSettings.targetTexture.Release();
            _document.panelSettings.targetTexture.Create();

            _root.Children().ToList().ForEach(t => t.Clear());
            _root.Clear();
        }

        public AtomMLVisualElement AddContainer(string name, Color backgroundColor, Vector2Int dimensions)
        {
            var parent = new AtomMLVisualElement();
            parent.name = name;

            parent.style.position = Position.Relative;
            parent.style.width = dimensions.x;
            parent.style.height = dimensions.y;
            parent.style.backgroundColor = backgroundColor;

            _root.Add(parent);

            return parent;
        }

        public AxisChart AddAxis(string name, Color backgroundColor, Vector2Int dimensions, VisualElement container = null)
        {
            var chart = new AxisChart(dimensions.x, dimensions.y);
            chart.name = name;
            chart.style.backgroundColor = backgroundColor;

            return AddChart(chart, container);
        }

        public SimpleLineChart Add_SimpleLine(double[,] matrice, float lineWidth, Vector2Int dimensions, VisualElement container = null)
        {
            var chart = new SimpleLineChart(matrice, lineWidth, dimensions.x, dimensions.y);

            return AddChart(chart, container);
        }

        public SimpleLineChart Add_SimpleLine(double[] points, float lineWidth, Vector2Int dimensions, VisualElement container = null)
        {
            var chart = new SimpleLineChart(points, lineWidth, dimensions.x, dimensions.y);

            return AddChart(chart, container);
        }

        public SimpleLineChart Add_SimpleLine(Func<List<Vector2>> getPoints, float lineWidth, Vector2Int dimensions, VisualElement container = null)
        {
            var chart = new SimpleLineChart(getPoints, lineWidth, dimensions.x, dimensions.y);
            
            return AddChart(chart, container);
        }

        public SimpleLineChart Add_SimpleLine(Func<List<double>> getPoints, float lineWidth, Vector2Int dimensions, VisualElement container = null)
        {
            var chart = new SimpleLineChart(getPoints, lineWidth, dimensions.x, dimensions.y);
            
            return AddChart(chart, container);
        }

        public Scatter2DChart Add_Scatter(double[,] matrice, VisualElement container = null)
        {
            var chart = new Scatter2DChart(() => matrice);
            return AddChart(chart, container);
        }

        public Scatter2DChart Add_Scatter(NVector[] array, VisualElement container = null)
        {
            var chart = new Scatter2DChart(() => NMatrix.DenseOfColumnVectors(array).Datas);
            return AddChart(chart, container);
        }

        public T AddChart<T>(T chart, VisualElement container) where T : AtomMLChart
        {
            if (container == null)
                _root.Add(chart);
            else container.Add(chart);

            chart.Refresh();
            return chart;
        }
    }
}
