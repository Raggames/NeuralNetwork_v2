using Sirenix.OdinInspector;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.PIDControllerTuning
{
    public class PIDControlledDroneGeneticTrainerManager : MonoBehaviour
    {
        [SerializeField] private PIDControlledDroneGeneticTrainer _droneGeneticTrainer;

        [Button]
        private async void TestFit()
        {
            await _droneGeneticTrainer.Fit();
        }
    }
}
