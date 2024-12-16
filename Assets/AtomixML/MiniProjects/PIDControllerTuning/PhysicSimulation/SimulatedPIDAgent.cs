using Sirenix.OdinInspector;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.PIDControllerTuning
{
    public class SimulatedPIDAgent : MonoBehaviour
    {
        private Rigidbody _rigidbody;

        [SerializeField] private Transform _target;
        [SerializeField] private float _forceMultiplier = 1;
        [SerializeField] private float _torqueMultiplier = 1;

        [FoldoutGroup("Translation"), SerializeField] private PIDFunction _xAxisPID;
        [FoldoutGroup("Translation"), SerializeField] private PIDFunction _yAxisPID;
        [FoldoutGroup("Translation"), SerializeField] private PIDFunction _zAxisPID;

        [FoldoutGroup("Rotation"), SerializeField] private PIDFunction _xRotAxisPID;
        [FoldoutGroup("Rotation"), SerializeField] private PIDFunction _yRotAxisPID;
        [FoldoutGroup("Rotation"), SerializeField] private PIDFunction _zRotAxisPID;


        private void Awake()
        {
            _rigidbody = GetComponent<Rigidbody>();

            _xAxisPID.Initialize(Time.fixedDeltaTime);
            _yAxisPID.Initialize(Time.fixedDeltaTime);
            _zAxisPID.Initialize(Time.fixedDeltaTime);
            _xRotAxisPID.Initialize(Time.fixedDeltaTime);
            _yRotAxisPID.Initialize(Time.fixedDeltaTime);
            _zRotAxisPID.Initialize(Time.fixedDeltaTime);
        }

        private void FixedUpdate()
        {
            var offset = transform.position - _target.position;

            var x_force = (float)_yAxisPID.Compute(offset.x, 0);
            var y_force = (float)_yAxisPID.Compute(offset.y, 0);
            var z_force = (float)_yAxisPID.Compute(offset.z, 0);

            _rigidbody.AddForce(new Vector3(x_force, y_force, z_force) * _forceMultiplier);

            var offsetR = transform.rotation.eulerAngles - _target.rotation.eulerAngles;

            var xr_force = (float)_xRotAxisPID.Compute(WrapAngle(offsetR.x), 0);
            var yr_force = (float)_yRotAxisPID.Compute(WrapAngle(offsetR.y), 0);
            var zr_force = (float)_zRotAxisPID.Compute(WrapAngle(offsetR.z), 0);

            //_rigidbody.AddForceAtPosition(Vector3.right, new Vector3(0, zr_force, 0) * _torqueMultiplier);
            _rigidbody.AddTorque(new Vector3(xr_force, yr_force,  zr_force) * _torqueMultiplier);
        }

        private float WrapAngle(float angle)
        {
            return angle > 180 ? angle - 360 : angle;
        }
    }

}
