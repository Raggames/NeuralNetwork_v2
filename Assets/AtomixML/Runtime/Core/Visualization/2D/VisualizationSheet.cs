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

        public SimpleLineChart Add_SimpleLineChart(double[,] matrice)
        {
            var chart = new SimpleLineChart(() => matrice);

            _root.Add(chart);


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
