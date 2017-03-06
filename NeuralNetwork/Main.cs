using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Printing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NeuralNetwork_C;
namespace NeuralNetwork {
	public partial class NeuralNetworkSystem : Form {
        NeuralNetworkC NN = null;
		int count = 0;
		bool FileLoaded;
		public NeuralNetworkSystem() {
			InitializeComponent();
			//Initilize the buttons
			btn_run.Enabled = false;
			btn_del.Enabled = false;
			btn_ins.Enabled = false;
			btn_reset.Enabled = false;

			FileLoaded = false;
		}

		//Import .cal File
		private void OpenFile_Click(object sender, EventArgs e) {
			//new a NeuralNetwork class
            NN = new NeuralNetworkC();
			
			//Initilize the buttons
			btn_run.Enabled = false;
			btn_del.Enabled = false;
			btn_ins.Enabled = false;
			btn_reset.Enabled = false;
			
			//if data is imported successful
			if(NN.ImportData()){
				NN_system.SelectedObject = NN;
				neuron_list.Items.Clear();

				//Show the hidden layers in list_box
				for(int i=1; i<NN.layer.Count-1; i++)
					neuron_list.Items.Add(NN.layer[i]);
				
				correctness_table.Rows.Clear();
				correctness_table.Columns.Clear();

				//Add the datagridview table of correctness examination
				for(int i=0; i<NN.Dimout; i++)
					correctness_table.Columns.Add("y"+i.ToString(), "y"+i.ToString());

				for(int i=0; i<NN.Dimout; i++){
					correctness_table.Rows.Add();
					correctness_table.Rows[i].HeaderCell.Value = "Out " + i.ToString();
				}

				correctness_table.RowHeadersWidth = 100;

				//Enable the reset button
				btn_reset.Enabled = true;
				FileLoaded = true;
				DrawNeuralDiagram();
				return;
			}
			else MessageBox.Show("Fail to import data!");
		}

		protected override bool ProcessCmdKey(ref Message msg, Keys keyData) {
            //Detect keypressed events. If ESC was pressed, terminate the application.
            if (keyData == Keys.Escape) {
                Close();
                return true;
            }
            return base.ProcessCmdKey(ref msg, keyData);
        }

		private void btn_run_Click(object sender, EventArgs e) {
			progressBar.Minimum = 0;
			progressBar.Maximum = NN.IterationLimit;
			progressBar.Value = 0;
			//Start training
			for(int i=0; i<NN.IterationLimit; i++){
				NN.UpdateNN();

				NN.Validation();
				chart.Series[0].Points.AddXY(count, NN.ErrInRMS);
				chart.Series[1].Points.AddXY(count, NN.ErrOutRMS);
				count++;
				UpdateInformation();
				//if stop_by_best is checked, the training will stop when NN System reach the current best correctness
				if(stop_by_best.Checked)
					if(NN.Correctness==NN.CurrentBestOutCorrectness){
						progressBar.Value = progressBar.Maximum;
						break;
					}
				progressBar.Value++;
			}			
		}

		private void UpdateInformation(){
			//Update the Error rate information
			string text = "";

			//The rms error of training sample
			text = "In-Sample RMS Error : " + Math.Round(NN.ErrInRMS,4);
			lbl_ein.Text = text;

			//The rms error of testing sample
			text = "Out-of-Sample RMS Error : " + Math.Round(NN.ErrOutRMS,4);
			lbl_eout.Text = text;
		
			//The 0/1 error of training sample
			text = "In-Sample 0/1 Error : " + Math.Round(1.0-NN.EinCorrectness,4)*100 + " %";
			lbl_errin01.Text = text;

			//The 0/1 error of testing sample
			text = "Out-of-Sample 0/1 Error : " + Math.Round(1.0-NN.Correctness,4)*100 + " %";
			lbl_err01.Text = text;

			//The current best 0/1 error of testing sample
			text = "Best-Out-of-Sample 0/1 Error : " + Math.Round(1.0-NN.CurrentBestOutCorrectness,4)*100 + " %";
			lbl_err01_best.Text = text;
			
			lbl_ein.Refresh();
			lbl_eout.Refresh();
			lbl_errin01.Refresh();
			lbl_err01.Refresh();
			lbl_err01_best.Refresh();
		}

		private void btn_test_Click(object sender, EventArgs e) {
			//Validate the NN system by test the testing data
			NN.Validation();
			int[][] table = NN.ValidationTable();
			for(int i=0; i<NN.Dimout; i++)
				for(int j=0; j<NN.Dimout; j++)
					correctness_table.Rows[i].Cells[j].Value = table[i][j];
			lbl_correctness.Text = "Correctness : " + (Math.Round(NN.Correctness,4)*100).ToString() + " %";
		}

		private void btn_ins_Click(object sender, EventArgs e) {
			//Insert a new layer
			int index = neuron_list.SelectedIndex;
			NN.InsertHiddenLayer(neuron_list.SelectedIndex+1, 5);
			
			//Update the list
			neuron_list.Items.Clear();
			for(int i=1; i<NN.layer.Count-1; i++)
				neuron_list.Items.Add(NN.layer[i]);
			neuron_list.SelectedIndex = index;

			DrawNeuralDiagram();
		}

		private void btn_del_Click(object sender, EventArgs e) {
			//Delete the selected layer
			if(neuron_list.SelectedIndex+1>=0 && neuron_list.SelectedIndex+1<NN.NumOfLayer-1)
				NN.DeleteHiddenLayer(neuron_list.SelectedIndex+1);
			
			//Update the list
			neuron_list.Items.Clear();
			for(int i=1; i<NN.layer.Count-1; i++)
				neuron_list.Items.Add(NN.layer[i]);

			DrawNeuralDiagram();
		}

		private void btn_reset_Click(object sender, EventArgs e) {
			//Reset NN system and chart
			NN.Reset();
			chart.Series[0].Points.Clear();
			chart.Series[1].Points.Clear();
			count = 0;
			btn_run.Enabled = true;
			btn_ins.Enabled = true;
			btn_del.Enabled = true;
		}

		private void neuron_list_SelectedIndexChanged(object sender, EventArgs e) {
			//Show number of neurons of selected layer
			num_perc.Value = NN.layer[neuron_list.SelectedIndex+1].InputDim-1;
		}
		
		private void num_perc_ValueChanged(object sender, EventArgs e) {
			int index = 0;
			//Add or delete a neuron in chosen layer
			if(neuron_list.SelectedIndex+1>=0 && neuron_list.SelectedIndex+1<NN.NumOfLayer-1){
				index = neuron_list.SelectedIndex+1;
				NN.EditHiddenLayer(index, (int)num_perc.Value);
				DrawNeuralDiagram();
			}

			//Update the list
			neuron_list.Items.Clear();
			for(int i=1; i<NN.layer.Count-1; i++){
				neuron_list.Items.Add(NN.layer[i]);
			}
			neuron_list.SelectedIndex = index-1;
		}

		private void btn_ins_MouseEnter(object sender, EventArgs e) {
			//Show the tips when the cursor enter the button
			ToolTip tips = new ToolTip();
			tips.ToolTipTitle = "Add hidden layer";
			tips.SetToolTip(btn_ins,"Add hidden layer with 5 neurons");
			tips.ToolTipIcon = ToolTipIcon.Info;
		}

		private void DrawNeuralDiagram() {
			Graphics g;
            g = panel.CreateGraphics();
            //Draw(g, panel.ClientRectangle);
		}

		void Draw(Graphics g, Rectangle bound){
			//Draw the Neural Network diagram
			g.Clear(Color.White);
			float[][][] points = new float[NN.NumOfLayer][][];
			
			//Initiate the array to save the coordinate of all points of neurons
			for(int i =0; i<NN.NumOfLayer; i++){
				points[i] = new float[NN.layer[i].InputDim][];
				for(int j =0; j<NN.layer[i].InputDim; j++)
					points[i][j] = new float[2];
			}
			
			//new a StringFormat
			StringFormat sf = new StringFormat();
            sf.Alignment = StringAlignment.Center;
            sf.LineAlignment = StringAlignment.Center;

			//Initialize neuron
            float unit = 1;
            Font dynamicFont;
            RectangleF neuron = RectangleF.Empty;
            neuron.Width = neuron.Height = unit * 2 / 3;            
            float start =　unit - neuron.Height / 2;
			neuron = Rectangle.Empty;

			//Calculate and save the corrdinates
            for( int i = 0; i < NN.NumOfLayer; i++){
				unit = bound.Height / ((float)NN.layer[i].InputDim);
				neuron.Width = neuron.Height = unit * 2 / 3;
				neuron.X = (bound.Width / (NN.NumOfLayer+1))*(i+1);
				dynamicFont = new Font("微軟正黑體", unit / 4, FontStyle.Bold | FontStyle.Italic);

				for( int j = 1; j < NN.layer[i].InputDim; j++){
					points[i][j][0] = neuron.X;
					points[i][j][1] = (j) * unit;
				}
			}

			PointF p1, p2;
			p1 = new Point();
			p2 = new Point();

			//Draw the line First
			for( int i = 0; i < NN.NumOfLayer-1; i++){				
				for( int j = 1; j < NN.layer[i].InputDim; j++){
					p1.X = points[i][j][0];
					p1.Y = points[i][j][1];
					for( int k = 1; k < NN.layer[i+1].InputDim; k++){
						p2.X = points[i+1][k][0];
						p2.Y = points[i+1][k][1];
						g.DrawLine(Pens.Red,p1, p2);
					}
				}
			}
			//Draw the circle fill with white color
			for( int i = 0; i < NN.NumOfLayer; i++){				
				unit = bound.Height / ((float)NN.layer[i].InputDim);
				neuron.Width = neuron.Height = ( unit * 2 / 3);
				dynamicFont = new Font("微軟正黑體", unit / 6, FontStyle.Bold | FontStyle.Italic);
				for( int j = 1; j < NN.layer[i].InputDim; j++){
					neuron.X = points[i][j][0] - neuron.Width/2;
					neuron.Y = points[i][j][1] - neuron.Height/2;
					g.FillEllipse(Brushes.White, neuron);
					g.DrawEllipse(Pens.Black, neuron);
					
					g.DrawString("x" + i.ToString() + j.ToString(),dynamicFont, Brushes.Blue, neuron, sf);
				}
			}
			/*
			float height = 30;
			float width  = bound.Width / (NN.NumOfLayer+1)/5*2;
			RectangleF weight = RectangleF.Empty;
			for( int i = 1; i < NN.NumOfLayer; i++){				
				unit = bound.Height / ((float)NN.layer[i].InputDim);
				neuron.Width = neuron.Height = ( unit * 2 / 3);
				dynamicFont = new Font("微軟正黑體", height/5, FontStyle.Bold | FontStyle.Italic);
				for( int j = 1; j < NN.layer[i].InputDim; j++){
					neuron.X = points[i][j][0] - neuron.Width/2;
					neuron.Y = points[i][j][1] - neuron.Height/2;
					g.FillRectangle(Brushes.White, neuron.X-width, neuron.Y+neuron.Height/2-height/2, width, height);
					g.DrawRectangle(Pens.Black, neuron.X-width, neuron.Y+neuron.Height/2-height/2, width, height);
					weight.X = neuron.X+neuron.Width;
					weight.Y = neuron.Y+neuron.Height/2-height/2;
					weight.Width = width;
					weight.Height = height;

					for( int k = 1; k < NN.layer[i].InputDim; k++);
						//g.DrawString(Math.Round(NN.layer[i].Weights[j-1][k-1],4).ToString(),dynamicFont, Brushes.Blue, weight, sf);
					
				}
			}*/
        }
		
		private void panel_Paint(object sender, PaintEventArgs e) {
			if(FileLoaded)
				DrawNeuralDiagram();
		}

		private void splitContainer1_SplitterMoved(object sender, SplitterEventArgs e) {
			if(FileLoaded)
				DrawNeuralDiagram();
		}

		private void splitContainer2_SplitterMoved(object sender, SplitterEventArgs e) {
			if(FileLoaded)
				DrawNeuralDiagram();
		}

	}
}
