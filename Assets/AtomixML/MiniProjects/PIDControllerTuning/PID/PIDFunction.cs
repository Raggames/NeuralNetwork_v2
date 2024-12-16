using Sirenix.OdinInspector;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.PIDControllerTuning
{
    [Serializable]
    public class PIDFunction
    {
        // tableau valeurs precedentes
        // delta t

        [SerializeField] private double _p  = .6;
        [SerializeField] private double _i = .45;
        [SerializeField] private double _d = .125;

        /// <summary>
        /// écart de temps entre deux valeurs
        /// </summary>
        [SerializeField] private double _deltaTime;

        [SerializeField] private double _maxRecordedTime = .33;
        [ShowInInspector, ReadOnly] private int _recordedValues = 0;
        [ShowInInspector, ReadOnly] private List<double> _samples = new List<double>();

        public List<double> samples => _samples;

        [Button]
        public void Initialize()
        {
            _recordedValues = (int)(_maxRecordedTime / _deltaTime);
        }

        public double Compute(double currentValue, double targetValue)
        {
            var current_error = targetValue - currentValue;
            var result = _p * current_error;

            if(_samples.Count > _recordedValues)
                _samples.RemoveAt(0);

            _samples.Add(currentValue);

            return result;
        }
    }
}
