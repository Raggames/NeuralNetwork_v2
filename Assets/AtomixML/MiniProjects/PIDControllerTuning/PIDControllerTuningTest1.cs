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
        [SerializeField] private int _maxDisplayedPoints = 500;

        [SerializeField, ReadOnly] private List<double> _samples = new List<double>();
        [SerializeField, ReadOnly] private List<double> _consignes = new List<double>();

        private SimpleLineChart _samplesLine;
        private SimpleLineChart _consigneLine;



        private void Start()
        {
            _visualizationSheet = GetComponentInChildren<VisualizationSheet>();

            var parent = _visualizationSheet.AddContainer("c1", Color.white, _graphDimensions);
            var axis = _visualizationSheet.AddAxis("a1", new Color(0, 0, 0, 0), _graphDimensions, parent);

            _samplesLine = _visualizationSheet.Add_SimpleLine(() => _samples, _graphLineWidth, _graphDimensions, parent);
            _samplesLine.backgroundColor = new Color(0, 0, 0, 0);

            _consigneLine = _visualizationSheet.Add_SimpleLine(() => _consignes, _graphLineWidth, _graphDimensions, parent);
            _consigneLine.strokeColor = Color.green;
            _consigneLine.backgroundColor = new Color(0, 0, 0, 0);

            _samplesLine.yRange = new Vector2(0, 30);
            _consigneLine.yRange = new Vector2(0, 30);

            _pidFunction.SetTime(Time.fixedDeltaTime);
        }

        private void FixedUpdate()
        {
            if(_samples.Count > _maxDisplayedPoints)
            {
                _samples.RemoveAt(0);
                _consignes.RemoveAt(0);
            }

            _samples.Add(_current);
            _consignes.Add(_consigne);

            _current += (float) _pidFunction.Compute(_current, _consigne);

            _samplesLine.Refresh();
            _consigneLine.Refresh();
        }
    }

}
