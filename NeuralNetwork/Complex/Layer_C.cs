using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Drawing;
using ComplexNum;
namespace NeuralNetwork_C{
	class Layer_C {
		Random rand = new Random();
		Complex[] neurons;
		Complex[] epsilons;
		Complex[][] weights;
		Complex[][] weights_prev;
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

		public Complex[] Neurons{
			get{ return neurons; }
			set{ neurons = value; }
		}

		public Complex[] Epsilons{
			get{ return epsilons; }
			set{ epsilons = value; }
		}

		public Complex[][] Weights{
			get{ return weights; }
		}

		public Layer_C(int input_dimension, int output_dimension){
			//Initiation should include new Complex();
			Dout = output_dimension;
			Din  = input_dimension;
			neurons = new Complex[input_dimension];
			epsilons    = new Complex[input_dimension];
			weights = new Complex[input_dimension][];
			weights_prev = new Complex[input_dimension][];
			for(int i=0; i<input_dimension; i++){
                neurons[i] = new Complex();
                epsilons[i] = new Complex();
				weights[i] = new Complex[output_dimension];
				weights_prev[i] = new Complex[output_dimension];
                for (int j = 0; j < output_dimension; j++)
                {
                    weights[i][j] = new Complex();
                    weights_prev[i][j] = new Complex();
                }
			}
			InitWeights();
		}

		public void Reset(){
			neurons = new Complex[Din];
			epsilons= new Complex[Din];
			weights = new Complex[Din][];
			weights_prev = new Complex[Din][];
            for (int i = 0; i < Din; i++)
            {
                neurons[i]  = new Complex();
                epsilons[i] = new Complex();

                weights[i] = new Complex[Dout];
                weights_prev[i] = new Complex[Dout];
                for (int j = 0; j < Dout; j++)
                {
                    weights[i][j] = new Complex();
                    weights_prev[i][j] = new Complex();
                }
			}
			InitWeights();
		}

		public void InitWeights(){
			//Bug, need to add one more dimension for x0
			//But we can do it on the input data before it send in;
			for(int i=0; i<Din; i++)
				for(int j=0; j<Dout; j++){
					weights[i][j].a = (rand.NextDouble()-0.5)*2.0;
					weights[i][j].b = (rand.NextDouble()-0.5)*2.0;
                    //weights[i][j].b = 0.0;
					weights_prev[i][j].a = weights_prev[i][j].b = 0.0;
				}
		}

		public Complex[] Output(bool last){
			Complex[] output = new Complex[Dout+1];
			Complex value = new Complex();
            output[0] = new Complex(1.0,0.0);
			for(int i=1; i<Dout+1; i++){
                output[i] = new Complex();
                value = new Complex();
                if (Din-1 == Dout)
                {
                    value = value + neurons[i-1] * weights[i-1][i - 1];
                    Console.WriteLine("Flat");
                }
                else
                {
                    for (int j = 0; j < Din; j++)
                    {
                        value = value + neurons[j] * weights[j][i - 1];
                    }
                }
				//Activation function maybe change to tanh();
                Complex N = new Complex(Math.Tanh(value.a), Math.Tan(value.b));
                Complex D = new Complex(1.0, Math.Tanh(value.a) * Math.Tan(value.b));
			    
				
                //value = N / D;
                
                //Activation function f(z) = {z/|z|}
                //value = value / value.Norm2();
                Complex R = new Complex(Math.Exp(-value.a),0.0);
                Complex I = new Complex(Math.Cos(-value.b), Math.Sin(-value.b));
                Complex one = new Complex(1.0, 0.0);
                /*if (!last)
                    value = one / (one + (R * I));
                else
                    ;*/
                //value.a = 1.0 / (1.0 + Math.Exp(-value.a));
                //value.b = 1.0 / (1.0 + Math.Exp(-value.b));
				output[i] = value;
			}
					
			return output;
		}

		public void UpdateLastLayerEpsilon(Complex[] y){
            Complex one = new Complex(1.0,0.0);
			for(int i=1; i<Din; i++){
                //epsilons[i] = -2.0 * (y[i - 1] - neurons[i]) * ((neurons[i]) * (one - neurons[i]));
                
                
                

                //epsilons[i].a = -2.0 * (y[i - 1].a - neurons[i].a) * neurons[i].a * (1.0 - neurons[i].a);
                //epsilons[i].b = -2.0 * (y[i - 1].b - neurons[i].b) * neurons[i].b * (1.0 - neurons[i].b);

                epsilons[i] = -2.0 * (y[i - 1] - neurons[i]);// *new Complex(1.0, 1.0);// *((neurons[i]) * (one - neurons[i]));
                //epsilons[i] = -2.0 * (y[i - 1] - neurons[i]) *((neurons[i]) * (one - neurons[i]));
                /*epsilons[i].a = 1.0;
                epsilons[i].b = 1.0;*/
			}
		}

		public void UpdateEpsilon(Complex[] e){
			for(int i=1; i<Din; i++){
				epsilons[i] = new Complex(0.0, 0.0);

				for(int j=0; j<Dout; j++)
					epsilons[i] = epsilons[i] + e[j+1]*weights[i][j];

                Complex one = new Complex(1.0, 0.0);
                Complex fRI = new Complex(neurons[i].a * (1.0 - neurons[i].a), neurons[i].b * (1.0 - neurons[i].b));
                //epsilons[i] = epsilons[i] * fRI;
                //epsilons[i] = epsilons[i] * neurons[i] * (one - neurons[i]) * new Complex(1.0, 1.0);
				//epsilons[i].a *= ((neurons[i].a)*(1.0-neurons[i].a));
				//epsilons[i].b *= ((neurons[i].b)*(1.0-neurons[i].b));
			}
		}

		public void UpdateWeight(Complex[] e, double eta, double alpha){
			Complex delta_weight = new Complex(0.0, 0.0);
			for(int i=0; i<Din; i++){
				//delta_weight = 0.0;
				delta_weight = new Complex(0.0, 0.0);
                Complex eta_C = new Complex(-eta, 0.0);
                Complex alpha_C = new Complex(alpha, 0.0);
				for(int j=0; j<Dout; j++){
                    //Important part
                    delta_weight = eta_C * e[j + 1] * neurons[i] + alpha_C * weights_prev[i][j];

					weights_prev[i][j] = delta_weight;
					weights[i][j] = weights[i][j] + delta_weight;
				}
			}

			Complex norm = new Complex(0.0, 0.0);
			//Max norm
			for(int j=0; j<Dout; j++){
				delta_weight = new Complex(0.0, 0.0);
				for(int i=0; i<Din; i++){
					norm = norm + weights[i][j]*weights[i][j];
				}
				double norm2 = Math.Sqrt(norm.a*norm.a+norm.b*norm.b);
				if(norm2 > 250.0){
					for(int i=0; i<Din; i++)
						weights[i][j] = weights[i][j] / new Complex(norm2, 0.0);
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
