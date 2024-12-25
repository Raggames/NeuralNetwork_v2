using Atom.MachineLearning.Core.Maths;
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

        [SerializeField] private bool _move = true;
        [SerializeField] private bool _rotate = false;

        [SerializeField] private float _targetMoveRange;
        [SerializeField] private float _targetSpeed;
        [SerializeField] private float _targetRotationTime;
        private Vector3 _targetCurrentPosition = Vector3.zero;
        private Quaternion _targetCurrentOrientationTarget;
        private Quaternion _targetBaseOrientation;
        private Vector3 _targetVelocity = Vector3.zero;
        private Vector3 _targetAngularVelocity = Vector3.zero;
        private float _rotationTimer = 0;

        private void Awake()
        {
            _targetCurrentPosition = transform.position;

        }

        private void Update()
        {
            if (_move)
            {
                var crt = (_target.position - _targetCurrentPosition).magnitude;
                if (crt < .01)
                {
                    _targetCurrentPosition = UnityEngine.Random.insideUnitSphere * _targetMoveRange + transform.position;
                }

                _target.position = Vector3.SmoothDamp(_target.position, _targetCurrentPosition, ref _targetVelocity, _targetSpeed);
            }

            if (_rotate)
            {
                if (_rotationTimer >= _targetRotationTime)
                {
                    _targetCurrentOrientationTarget = Quaternion.LookRotation(new Vector3(
                        MLRandom.Shared.Range(-180, 180),
                        MLRandom.Shared.Range(-180, 180),
                        MLRandom.Shared.Range(-180, 180)), Vector3.up);

                    _targetBaseOrientation = _target.rotation;
                    _rotationTimer = 0;
                }

                _rotationTimer += Time.deltaTime;
                _target.rotation = Quaternion.Slerp(_targetBaseOrientation, _targetCurrentOrientationTarget, _rotationTimer / _targetRotationTime);
            }


        }
    }
}
