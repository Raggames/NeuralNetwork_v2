using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Atom.MachineLearning.Core.Maths
{
    public class MLMetricFunctions
    {
        /// <summary>
        /// Pearson correlation coefficient helps us measure the similarity between two sets of data. 
        /// At its core, it is the normalized covariance between two variables- the ratio between their covariance and the product of their standard deviations.
        /// </summary>
        /// <param name="t_datas"></param>
        /// <param name="o_datas"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public static double PearsonCoefficient(NVector[] t_datas, NVector[] o_datas)
        {
            if (t_datas.Length != o_datas.Length || t_datas[0].length != o_datas[0].length) throw new Exception($"Matrix sizes must match");

            int dimension = o_datas[0].length;

            double sum_covariance = 0.0;
            double t_std = 0.0;
            double o_std = 0.0;

            double[] t_means = new double[t_datas.Length];
            for (int i = 0; i < t_datas.Length; i++)
                t_means[i] = t_datas[i].Average();

            double[] o_means = new double[o_datas.Length];
            for (int i = 0; i < o_datas.Length; i++)
                o_means[i] = o_datas[i].Average();

            for (int i = 0; i < t_datas.Length; i++)
            {
                for (int j = 0; j < o_datas.Length; j++)
                {
                    for (int k = 0; k < dimension; ++k)
                    {
                        sum_covariance += (t_datas[i][k] - t_means[i]) * (o_datas[i][k] - o_means[i]);
                        t_std += Math.Pow((t_datas[i][k] - t_means[i]), 2);
                        o_std += Math.Pow((o_datas[i][k] - o_means[i]), 2);
                    }
                }
            }

            double product_stds = Math.Sqrt(t_std) * Math.Sqrt(o_std);
            double result = sum_covariance / product_stds;
            return result;
        }

        /// <summary>
        /// The coefficient of determination, otherwise known as R², is a very handy and useful metric for regression-type problems.
        /// </summary>
        /// <returns></returns>
        public static double RR(NVector[] t_datas, NVector[] o_datas)
        {
            if (t_datas.Length != o_datas.Length || t_datas[0].length != o_datas[0].length) throw new Exception($"Matrix sizes must match");

            int dimension = o_datas[0].length;

            /*
             R² = 1 − ∑(yi − ŷi)² / ∑(yi − ȳ)²
             yi représente les valeurs observées de la variable dépendante,
             ŷi représente les valeurs prédites par le modèle,
             ȳ est la moyenne des valeurs observées,
             ∑(yi − ŷi)² est la somme des carrés des résidus (ou erreurs),
             ∑(yi − ȳ)² est la somme totale des carrés, qui mesure la dispersion totale des valeurs observées.
             */

            double[] t_means = new double[t_datas.Length];
            for (int i = 0; i < t_datas.Length; i++)
                t_means[i] = t_datas[i].Average();

            double sum_sqr_error = 0.0;
            double sum_dispersion = 0.0;

            for (int i = 0; i < t_datas.Length; i++)
            {
                for (int k = 0; k < dimension; ++k)
                {
                    sum_sqr_error += Math.Pow(t_datas[i][k] - o_datas[i][k], 2);
                    sum_dispersion += Math.Pow(t_datas[i][k] - t_means[i], 2);
                }
            }

            if (sum_dispersion == 0)
                return 0;

            double result = 1.0 - (sum_sqr_error / sum_dispersion);
            return result;
        }
    }
}
