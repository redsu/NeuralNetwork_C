using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Drawing;
namespace NeuralNetwork {
	class Layer {
		Random rand = new Random();
		double[] neurons;
		double[] epsilons;
		double[][] weights;
		double[][] weights_prev;
		int Dout, Din;
		int node = 0;

		public int Node{
			get{return node;}
			set{node = value;}
		}

		public int InputDim{
			get{ return Din; }
			set{ Din = value>0 ? value : Din; }
		}

		public int OutputDim{
			get{ return Dout; }
			set{ Dout = value>0 ? value : Dout; }
		}

		public double[] Neurons{
			get{ return neurons; }
			set{ neurons = value; }
		}

		public double[] Epsilons{
			get{ return epsilons; }
			set{ epsilons = value; }
		}

		public double[][] Weights{
			get{ return weights; }
		}

		public Layer(int input_dimension, int output_dimension){
			Dout = output_dimension;
			Din  = input_dimension;
			neurons = new double[input_dimension];
			epsilons    = new double[input_dimension];
			weights = new double[input_dimension][];
			weights_prev = new double[input_dimension][];
			for(int i=0; i<input_dimension; i++){
				weights[i] = new double[output_dimension];
				weights_prev[i] = new double[output_dimension];
			}
			InitWeights();
		}

		public void Reset(){
			neurons = new double[Din];
			epsilons    = new double[Din];
			weights = new double[Din][];
			weights_prev = new double[Din][];
			for(int i=0; i<Din; i++){
				weights[i] = new double[Dout];
				weights_prev[i] = new double[Dout];
			}
			InitWeights();
		}

		public void InitWeights(){
			//Bug, need to add one more dimension for x0
			//But we can do it on the input data before it send in;
			for(int i=0; i<Din; i++)
				for(int j=0; j<Dout; j++){
					weights[i][j] = (rand.NextDouble()-0.5)*2;
					weights_prev[i][j] = 0.0;
				}
		}

		public double[] Output(){
			double[] output = new double[Dout+1];
			double value;
			for(int i=1; i<Dout+1; i++){
				value = 0;

				for(int j=0; j<Din; j++)
					value += neurons[j] * weights[j][i-1];
				
				value = 1.0 / (1.0 + Math.Exp(-value));
				output[i] = value;
			}
			output[0] = 1.0;
					
			return output;
		}

		public void UpdateLastLayerEpsilon(double[] y){
			for(int i=1; i<Din; i++)
				epsilons[i] = -2 * (y[i-1] - neurons[i])*((neurons[i])*(1.0-neurons[i]));
		}

		public void UpdateEpsilon(double[] e){
			for(int i=1; i<Din; i++){
				epsilons[i] = 0;

				for(int j=0; j<Dout; j++)
					epsilons[i] += e[j+1]*weights[i][j];

				epsilons[i] *= ((neurons[i])*(1.0-neurons[i]));
			}
		}

		public void UpdateWeight(double[] e, double eta, double alpha){
			double delta_weight = 0.0;
			for(int i=0; i<Din; i++){
				delta_weight = 0.0;
				for(int j=0; j<Dout; j++){
					delta_weight = -eta * e[j+1] * neurons[i] + alpha * weights_prev[i][j];
					weights_prev[i][j] = delta_weight;
					weights[i][j] += delta_weight;
				}
			}

			double norm = 0.0;
			//Max norm
			for(int j=0; j<Dout; j++){
				delta_weight = 0.0;
				for(int i=0; i<Din; i++){
					norm += weights[i][j]*weights[i][j];
				}
				norm = Math.Sqrt(norm);
				if(norm > 150.0){
					for(int i=0; i<Din; i++)
						weights[i][j]/=norm;
					Console.WriteLine("over");
				}
			}
		}

		public void ShowWeights(int d, object sender){			
			for(int i=0; i<Dout; i++)
				for(int j=0; j<Din; j++)
					((ListBox)sender).Items.Add(String.Format("w[{0}][{2}][{1}] = {3:F4}",d,j,i,weights[j][i]));
		}

		public override string ToString() {
			return String.Format("Layer {0,3} : {1,3}  neurons", node, Din-1);
		}
	}
}
