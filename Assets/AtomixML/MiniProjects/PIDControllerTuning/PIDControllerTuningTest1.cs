using Atom.MachineLearning.Core.Visualization;
using Atom.MachineLearning.Core.Visualization.VisualElements;
using Sirenix.OdinInspector;
using System.Collections.Generic;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.PIDControllerTuning
{
    public class PIDControllerTuningTest1 : MonoBehaviour
    {
        [SerializeField] private VisualizationSheet _visualizationSheet;

        [SerializeField] private PIDFunction _pidFunction;

        [SerializeField] private float _consigne = 10;
        [SerializeField] private float _current = 0;

        [Header("Graph")]
        [SerializeField] private Vector2Int _graphDimensions = new Vector2Int(600, 400);
        [SerializeField] private float _graphLineWidth = 2.5f;


        [SerializeField, ReadOnly] private List<double> _samples = new List<double>();
        [SerializeField, ReadOnly] private List<double> _consignes = new List<double>();

        private SimpleLineChart _samplesLine;
        private SimpleLineChart _consigneLine;

        private void Awake()
        {
            _visualizationSheet = GetComponentInChildren<VisualizationSheet>();
            _samplesLine = _visualizationSheet.Add_SimpleLine(() => _samples, _graphLineWidth, _graphDimensions);
           // _samplesLine.backgroundColor = new Color(0, 0, 0, 0);
            _consigneLine = _visualizationSheet.Add_SimpleLine(() => _consignes, _graphLineWidth, _graphDimensions);
            //_consigneLine.strokeColor = Color.green;
        }

        private void Update()
        {
            _samples.Add(_current);
            _consignes.Add(_consigne);

            _current += (float) _pidFunction.Compute(_current, _consigne);

            _samplesLine.Refresh();
            _consigneLine.Refresh();
        }
    }

}
