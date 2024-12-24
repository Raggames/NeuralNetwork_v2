using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.PIDControllerTuning
{
    public class TargetMover : MonoBehaviour
    {
        [SerializeField] private Transform _target;

        [SerializeField] private float _targetMoveRange;
        [SerializeField] private float _targetSpeed;
        private Vector3 _targetCurrentPosition = Vector3.zero;
        private Vector3 _targetVelocity = Vector3.zero;

        private void Awake()
        {
            _targetCurrentPosition = transform.position;

        }

        private void Update()
        {
            var crt = (_target.position - _targetCurrentPosition).magnitude;
            if (crt < .01)
            {
                _targetCurrentPosition = UnityEngine.Random.insideUnitSphere * _targetMoveRange + transform.position;
            }

            _target.position = Vector3.SmoothDamp(_target.position, _targetCurrentPosition, ref _targetVelocity, _targetSpeed);

        }
    }
}
