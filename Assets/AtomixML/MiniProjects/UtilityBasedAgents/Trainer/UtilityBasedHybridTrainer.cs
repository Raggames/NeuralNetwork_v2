using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Atom.MachineLearning.MiniProjects.UtilityBasedAgents
{
    /// <summary>
    /// An hybrid trainer using genetic and reinforced learning techniques to learn best utility functions (linear and exponentials) to 
    /// output the best possible competitive agents for complex actions/behaviours.
    /// 
    /// The trainer is a kind of hierarchical reinforced algorithm as we will use utility functions to rank the best action for any situation.
    /// </summary>
    public class UtilityBasedHybridTrainer : MonoBehaviour
    {
        [SerializeField] private AIAgent _pf_agent;

        [Header("Genetic Parameters")]
        [SerializeField] private int _agentsCount = 50;

        /*
         There is two major axis to the approach of this training algorithm.
         We have complex behaviours that would be intractable to describe with simple Reward Functions like in classic reinforcement learning. 
        Those behaviours would profit more from the genetic/competitive/evolutionnary approach
        Nevertheless, there is situations where we know with a higher confidence what the agent should or shouldn't have done.

        The goal of this experiment is to try to use a reinforced approach coupled to the genetic algorithm to help fitting the actions.
        For instance, we know that if an agent is not using its TryEatAction whereas his life points goes low / hunger goes high, we will be able to penalize it.
        On the other hand we will reward usage of the eat action in predetermined parametered situations.

        In the
         
         */
    }
}
