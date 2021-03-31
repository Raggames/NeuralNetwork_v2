using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assets.Job_NeuralNetwork.Scripts.GeneticNetwork
{
    [Serializable]
    public struct Gene
    {
        public int MutationVersion;
        public string TraitName;

        public float Value;

        public float Dominance; // Recessivity

        public Gene(int mutationverion, string traitName, float value, float dominance)
        {
            this.MutationVersion = mutationverion;
            this.TraitName = traitName;
            this.Value = value;
            this.Dominance = dominance;
        }
    }
}
