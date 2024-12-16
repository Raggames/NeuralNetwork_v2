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

        /// <summary>
        /// Derivative is approx by the line from n to n-_derivativeRange
        /// </summary>
        [SerializeField] private int _derivativeRange = 10;

        [ShowInInspector, ReadOnly] private int _recordedValues = 0;
        [ShowInInspector, ReadOnly] private List<double> _samples = new List<double>();
        [ShowInInspector, ReadOnly] private List<double> _targets = new List<double>();

        public List<double> samples => _samples;
        public List<double> targets => _targets;

        [Button]
        public void Initialize(float dtime)
        {
            _deltaTime = dtime;
            _recordedValues = (int)(_maxRecordedTime / _deltaTime);
        }

        public double Compute(double currentValue, double targetValue)
        {
            if (_samples.Count > _recordedValues)
            {
                _samples.RemoveAt(0);
                _targets.RemoveAt(0);
            }

            _samples.Add(currentValue);
            _targets.Add(targetValue);

            var current_error = targetValue - currentValue;
            var result = _p * current_error;

            result += _i * Integral();
            result += _d * Derivative(current_error);

            return result;
        }

        private double Integral()
        {
            double sum = 0.0;
            for(int i = 0; i < _samples.Count; i++)
            {
                sum += _deltaTime * (_targets[i] - _samples[i]);
            }

            return sum;
        }

        private double Derivative(double error)
        {
            if (_samples.Count <= 1)
                return 0;

            if(_samples.Count > _derivativeRange)
            {
                var y_offset = _samples[_samples.Count - 1] - _samples[_samples.Count - _derivativeRange];
                var x_offset = (_samples.Count - 1) * _deltaTime - (_samples.Count - _derivativeRange) * _deltaTime;

                var gradient  = y_offset / x_offset;

                return gradient * error;
            }
            else
            {
                var y_offset = _samples[_samples.Count - 1] - _samples[0];
                var x_offset = (_samples.Count - 1) * _deltaTime;

                var gradient = y_offset / x_offset;

                return gradient * error;
            }
        }
    }
}
