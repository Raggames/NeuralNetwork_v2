using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.PIDControllerTuning
{
    public class SimulatedPIDAgent : MonoBehaviour
    {
        private Rigidbody _rigidbody;

        [SerializeField] private Transform _target;
        [SerializeField] private PIDFunction _xAxisPID;
        [SerializeField] private PIDFunction _yAxisPID;
        [SerializeField] private PIDFunction _zAxisPID;
        [SerializeField] private float _forceMultiplier = 1;

        private void Awake()
        {
            _rigidbody = GetComponent<Rigidbody>();

            _xAxisPID.Initialize(Time.fixedDeltaTime);
            _yAxisPID.Initialize(Time.fixedDeltaTime);
            _zAxisPID.Initialize(Time.fixedDeltaTime);
        }

        private void FixedUpdate()
        {
            var offset = transform.position - _target.position;

            var x_force = (float)_yAxisPID.Compute(offset.x, 0);
            var y_force = (float)_yAxisPID.Compute(offset.y, 0);
            var z_force = (float)_yAxisPID.Compute(offset.z, 0);

            _rigidbody.AddForce(new Vector3(x_force, y_force, z_force) * _forceMultiplier);
        }
    }

}
