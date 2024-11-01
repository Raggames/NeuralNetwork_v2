using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;


namespace Atom.MachineLearning.Core
{
    /// <summary>
    /// Abstraction d'un trainer
    /// Un modèle/algorithme va toujours de paire avec son ITrainer, qui implémente toutes les fonctions d'apprentissage
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IMLTrainer<TModel, TModelInputData, TModelOutputData>
        where TModelInputData : IMLInOutData
        where TModelOutputData : IMLInOutData
        where TModel : IMLModel<TModelInputData, TModelOutputData>
    {
        public Task<ITrainingResult> Fit(TModel model, TModelInputData[] x_datas);
    }

    /// <summary>
    /// Abstraction d'un trainer
    /// Un modèle/algorithme va toujours de paire avec son ITrainer, qui implémente toutes les fonctions d'apprentissage
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IMLSupervisedTrainer<TModel, TModelInputData, TModelOutputData>
        where TModelInputData : IMLInOutData
        where TModelOutputData : IMLInOutData
        where TModel : IMLModel<TModelInputData, TModelOutputData>
    {
        public Task<ITrainingResult> Fit(TModel model, TModelInputData[] x_datas, TModelOutputData[] t_datas);
    }
}
