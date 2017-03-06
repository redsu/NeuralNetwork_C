namespace NeuralNetwork
{
	partial class NeuralNetworkSystem
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea1 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend1 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series1 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.Series series2 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(NeuralNetworkSystem));
            this.neuron_list = new System.Windows.Forms.ListBox();
            this.btn_run = new System.Windows.Forms.Button();
            this.chart = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.NN_system = new System.Windows.Forms.PropertyGrid();
            this.lbl_ein = new System.Windows.Forms.Label();
            this.lbl_eout = new System.Windows.Forms.Label();
            this.btn_del = new System.Windows.Forms.Button();
            this.lbl_err01 = new System.Windows.Forms.Label();
            this.tab = new System.Windows.Forms.TabControl();
            this.page_main = new System.Windows.Forms.TabPage();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.splitContainer2 = new System.Windows.Forms.SplitContainer();
            this.panel = new System.Windows.Forms.Panel();
            this.splitContainer3 = new System.Windows.Forms.SplitContainer();
            this.lbl_errin01 = new System.Windows.Forms.Label();
            this.stop_by_best = new System.Windows.Forms.CheckBox();
            this.lbl_err01_best = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.num_perc = new System.Windows.Forms.NumericUpDown();
            this.btn_ins = new System.Windows.Forms.Button();
            this.btn_reset = new System.Windows.Forms.Button();
            this.page_result = new System.Windows.Forms.TabPage();
            this.lbl_correctness = new System.Windows.Forms.Label();
            this.btn_test = new System.Windows.Forms.Button();
            this.correctness_table = new System.Windows.Forms.DataGridView();
            this.progressBar = new System.Windows.Forms.ProgressBar();
            this.OpenFile = new System.Windows.Forms.ToolStripButton();
            this.toolStrip = new System.Windows.Forms.ToolStrip();
            ((System.ComponentModel.ISupportInitialize)(this.chart)).BeginInit();
            this.tab.SuspendLayout();
            this.page_main.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer2)).BeginInit();
            this.splitContainer2.Panel1.SuspendLayout();
            this.splitContainer2.Panel2.SuspendLayout();
            this.splitContainer2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer3)).BeginInit();
            this.splitContainer3.Panel1.SuspendLayout();
            this.splitContainer3.Panel2.SuspendLayout();
            this.splitContainer3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.num_perc)).BeginInit();
            this.page_result.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.correctness_table)).BeginInit();
            this.toolStrip.SuspendLayout();
            this.SuspendLayout();
            // 
            // neuron_list
            // 
            this.neuron_list.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.neuron_list.FormattingEnabled = true;
            this.neuron_list.ItemHeight = 19;
            this.neuron_list.Location = new System.Drawing.Point(0, 0);
            this.neuron_list.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.neuron_list.Name = "neuron_list";
            this.neuron_list.Size = new System.Drawing.Size(235, 213);
            this.neuron_list.TabIndex = 0;
            this.neuron_list.SelectedIndexChanged += new System.EventHandler(this.neuron_list_SelectedIndexChanged);
            // 
            // btn_run
            // 
            this.btn_run.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btn_run.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btn_run.ForeColor = System.Drawing.Color.DarkGreen;
            this.btn_run.Location = new System.Drawing.Point(2, 44);
            this.btn_run.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.btn_run.Name = "btn_run";
            this.btn_run.Size = new System.Drawing.Size(74, 35);
            this.btn_run.TabIndex = 2;
            this.btn_run.Text = "Run";
            this.btn_run.UseVisualStyleBackColor = true;
            this.btn_run.Click += new System.EventHandler(this.btn_run_Click);
            // 
            // chart
            // 
            chartArea1.Name = "ChartArea1";
            this.chart.ChartAreas.Add(chartArea1);
            this.chart.Dock = System.Windows.Forms.DockStyle.Fill;
            legend1.Docking = System.Windows.Forms.DataVisualization.Charting.Docking.Top;
            legend1.Name = "Legend1";
            this.chart.Legends.Add(legend1);
            this.chart.Location = new System.Drawing.Point(0, 0);
            this.chart.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.chart.Name = "chart";
            series1.ChartArea = "ChartArea1";
            series1.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            series1.Legend = "Legend1";
            series1.Name = "In-Sample Error";
            series2.ChartArea = "ChartArea1";
            series2.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            series2.Legend = "Legend1";
            series2.Name = "Out-Of-Sample Error";
            this.chart.Series.Add(series1);
            this.chart.Series.Add(series2);
            this.chart.Size = new System.Drawing.Size(330, 202);
            this.chart.TabIndex = 3;
            this.chart.Text = "chart1";
            // 
            // NN_system
            // 
            this.NN_system.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.NN_system.Location = new System.Drawing.Point(6, 290);
            this.NN_system.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.NN_system.Name = "NN_system";
            this.NN_system.Size = new System.Drawing.Size(308, 209);
            this.NN_system.TabIndex = 4;
            // 
            // lbl_ein
            // 
            this.lbl_ein.AutoSize = true;
            this.lbl_ein.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.lbl_ein.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lbl_ein.ForeColor = System.Drawing.Color.Purple;
            this.lbl_ein.Location = new System.Drawing.Point(2, 139);
            this.lbl_ein.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.lbl_ein.Name = "lbl_ein";
            this.lbl_ein.Size = new System.Drawing.Size(153, 19);
            this.lbl_ein.TabIndex = 6;
            this.lbl_ein.Text = "In-Sample RMS Error : ";
            // 
            // lbl_eout
            // 
            this.lbl_eout.AutoSize = true;
            this.lbl_eout.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.lbl_eout.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lbl_eout.ForeColor = System.Drawing.Color.Purple;
            this.lbl_eout.Location = new System.Drawing.Point(2, 158);
            this.lbl_eout.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.lbl_eout.Name = "lbl_eout";
            this.lbl_eout.Size = new System.Drawing.Size(183, 19);
            this.lbl_eout.TabIndex = 7;
            this.lbl_eout.Text = "Out-of-Sample RMS Error : ";
            // 
            // btn_del
            // 
            this.btn_del.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btn_del.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btn_del.ForeColor = System.Drawing.Color.Navy;
            this.btn_del.Location = new System.Drawing.Point(89, 3);
            this.btn_del.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.btn_del.Name = "btn_del";
            this.btn_del.Size = new System.Drawing.Size(74, 35);
            this.btn_del.TabIndex = 10;
            this.btn_del.Text = "Delete";
            this.btn_del.UseVisualStyleBackColor = true;
            this.btn_del.Click += new System.EventHandler(this.btn_del_Click);
            // 
            // lbl_err01
            // 
            this.lbl_err01.AutoSize = true;
            this.lbl_err01.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.lbl_err01.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lbl_err01.ForeColor = System.Drawing.Color.DarkGreen;
            this.lbl_err01.Location = new System.Drawing.Point(2, 205);
            this.lbl_err01.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.lbl_err01.Name = "lbl_err01";
            this.lbl_err01.Size = new System.Drawing.Size(176, 19);
            this.lbl_err01.TabIndex = 11;
            this.lbl_err01.Text = "Out-of-Sample 0/1 Error : ";
            // 
            // tab
            // 
            this.tab.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tab.Controls.Add(this.page_main);
            this.tab.Controls.Add(this.page_result);
            this.tab.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tab.Location = new System.Drawing.Point(0, 29);
            this.tab.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.tab.Name = "tab";
            this.tab.SelectedIndex = 0;
            this.tab.Size = new System.Drawing.Size(899, 543);
            this.tab.TabIndex = 12;
            // 
            // page_main
            // 
            this.page_main.BackColor = System.Drawing.Color.LightBlue;
            this.page_main.Controls.Add(this.splitContainer1);
            this.page_main.Location = new System.Drawing.Point(4, 28);
            this.page_main.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.page_main.Name = "page_main";
            this.page_main.Padding = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.page_main.Size = new System.Drawing.Size(891, 511);
            this.page_main.TabIndex = 0;
            this.page_main.Text = "Neural Network";
            // 
            // splitContainer1
            // 
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.Location = new System.Drawing.Point(2, 3);
            this.splitContainer1.Name = "splitContainer1";
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.BackColor = System.Drawing.SystemColors.ControlLight;
            this.splitContainer1.Panel1.Controls.Add(this.splitContainer2);
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.BackColor = System.Drawing.SystemColors.ControlLight;
            this.splitContainer1.Panel2.Controls.Add(this.lbl_errin01);
            this.splitContainer1.Panel2.Controls.Add(this.stop_by_best);
            this.splitContainer1.Panel2.Controls.Add(this.lbl_err01_best);
            this.splitContainer1.Panel2.Controls.Add(this.label1);
            this.splitContainer1.Panel2.Controls.Add(this.num_perc);
            this.splitContainer1.Panel2.Controls.Add(this.NN_system);
            this.splitContainer1.Panel2.Controls.Add(this.btn_ins);
            this.splitContainer1.Panel2.Controls.Add(this.btn_reset);
            this.splitContainer1.Panel2.Controls.Add(this.btn_del);
            this.splitContainer1.Panel2.Controls.Add(this.lbl_ein);
            this.splitContainer1.Panel2.Controls.Add(this.lbl_err01);
            this.splitContainer1.Panel2.Controls.Add(this.lbl_eout);
            this.splitContainer1.Panel2.Controls.Add(this.btn_run);
            this.splitContainer1.Size = new System.Drawing.Size(887, 505);
            this.splitContainer1.SplitterDistance = 569;
            this.splitContainer1.TabIndex = 13;
            this.splitContainer1.SplitterMoved += new System.Windows.Forms.SplitterEventHandler(this.splitContainer1_SplitterMoved);
            // 
            // splitContainer2
            // 
            this.splitContainer2.BackColor = System.Drawing.Color.LightBlue;
            this.splitContainer2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer2.Location = new System.Drawing.Point(0, 0);
            this.splitContainer2.Name = "splitContainer2";
            this.splitContainer2.Orientation = System.Windows.Forms.Orientation.Horizontal;
            // 
            // splitContainer2.Panel1
            // 
            this.splitContainer2.Panel1.Controls.Add(this.panel);
            // 
            // splitContainer2.Panel2
            // 
            this.splitContainer2.Panel2.Controls.Add(this.splitContainer3);
            this.splitContainer2.Size = new System.Drawing.Size(569, 505);
            this.splitContainer2.SplitterDistance = 299;
            this.splitContainer2.TabIndex = 4;
            this.splitContainer2.SplitterMoved += new System.Windows.Forms.SplitterEventHandler(this.splitContainer2_SplitterMoved);
            // 
            // panel
            // 
            this.panel.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel.BackColor = System.Drawing.SystemColors.ControlLightLight;
            this.panel.Location = new System.Drawing.Point(0, -3);
            this.panel.Name = "panel";
            this.panel.Size = new System.Drawing.Size(569, 300);
            this.panel.TabIndex = 1;
            this.panel.Paint += new System.Windows.Forms.PaintEventHandler(this.panel_Paint);
            // 
            // splitContainer3
            // 
            this.splitContainer3.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer3.Location = new System.Drawing.Point(0, 0);
            this.splitContainer3.Name = "splitContainer3";
            // 
            // splitContainer3.Panel1
            // 
            this.splitContainer3.Panel1.Controls.Add(this.chart);
            // 
            // splitContainer3.Panel2
            // 
            this.splitContainer3.Panel2.Controls.Add(this.neuron_list);
            this.splitContainer3.Size = new System.Drawing.Size(569, 202);
            this.splitContainer3.SplitterDistance = 330;
            this.splitContainer3.TabIndex = 0;
            // 
            // lbl_errin01
            // 
            this.lbl_errin01.AutoSize = true;
            this.lbl_errin01.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.lbl_errin01.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lbl_errin01.ForeColor = System.Drawing.Color.DarkGreen;
            this.lbl_errin01.Location = new System.Drawing.Point(2, 186);
            this.lbl_errin01.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.lbl_errin01.Name = "lbl_errin01";
            this.lbl_errin01.Size = new System.Drawing.Size(146, 19);
            this.lbl_errin01.TabIndex = 18;
            this.lbl_errin01.Text = "In-Sample 0/1 Error : ";
            // 
            // stop_by_best
            // 
            this.stop_by_best.AutoSize = true;
            this.stop_by_best.ForeColor = System.Drawing.Color.Blue;
            this.stop_by_best.Location = new System.Drawing.Point(6, 261);
            this.stop_by_best.Name = "stop_by_best";
            this.stop_by_best.Size = new System.Drawing.Size(207, 23);
            this.stop_by_best.TabIndex = 17;
            this.stop_by_best.Text = "Stop When Found New Best";
            this.stop_by_best.UseVisualStyleBackColor = false;
            // 
            // lbl_err01_best
            // 
            this.lbl_err01_best.AutoSize = true;
            this.lbl_err01_best.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.lbl_err01_best.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lbl_err01_best.ForeColor = System.Drawing.Color.DarkRed;
            this.lbl_err01_best.Location = new System.Drawing.Point(2, 230);
            this.lbl_err01_best.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.lbl_err01_best.Name = "lbl_err01_best";
            this.lbl_err01_best.Size = new System.Drawing.Size(210, 19);
            this.lbl_err01_best.TabIndex = 16;
            this.lbl_err01_best.Text = "Best-Out-of-Sample 0/1 Error : ";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.label1.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.ForeColor = System.Drawing.Color.Crimson;
            this.label1.Location = new System.Drawing.Point(2, 93);
            this.label1.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(161, 19);
            this.label1.TabIndex = 15;
            this.label1.Text = "Number of perceptons :";
            // 
            // num_perc
            // 
            this.num_perc.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.num_perc.Location = new System.Drawing.Point(167, 91);
            this.num_perc.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.num_perc.Maximum = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            this.num_perc.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.num_perc.Name = "num_perc";
            this.num_perc.Size = new System.Drawing.Size(58, 27);
            this.num_perc.TabIndex = 14;
            this.num_perc.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.num_perc.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.num_perc.ValueChanged += new System.EventHandler(this.num_perc_ValueChanged);
            // 
            // btn_ins
            // 
            this.btn_ins.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btn_ins.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btn_ins.ForeColor = System.Drawing.Color.Navy;
            this.btn_ins.Location = new System.Drawing.Point(89, 44);
            this.btn_ins.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.btn_ins.Name = "btn_ins";
            this.btn_ins.Size = new System.Drawing.Size(74, 35);
            this.btn_ins.TabIndex = 8;
            this.btn_ins.Text = "Insert";
            this.btn_ins.UseVisualStyleBackColor = true;
            this.btn_ins.Click += new System.EventHandler(this.btn_ins_Click);
            this.btn_ins.MouseEnter += new System.EventHandler(this.btn_ins_MouseEnter);
            // 
            // btn_reset
            // 
            this.btn_reset.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btn_reset.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btn_reset.ForeColor = System.Drawing.Color.DarkRed;
            this.btn_reset.Location = new System.Drawing.Point(2, 3);
            this.btn_reset.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.btn_reset.Name = "btn_reset";
            this.btn_reset.Size = new System.Drawing.Size(74, 35);
            this.btn_reset.TabIndex = 12;
            this.btn_reset.Text = "Reset";
            this.btn_reset.UseVisualStyleBackColor = true;
            this.btn_reset.Click += new System.EventHandler(this.btn_reset_Click);
            // 
            // page_result
            // 
            this.page_result.Controls.Add(this.lbl_correctness);
            this.page_result.Controls.Add(this.btn_test);
            this.page_result.Controls.Add(this.correctness_table);
            this.page_result.Location = new System.Drawing.Point(4, 28);
            this.page_result.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.page_result.Name = "page_result";
            this.page_result.Padding = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.page_result.Size = new System.Drawing.Size(891, 511);
            this.page_result.TabIndex = 1;
            this.page_result.Text = "Test Result";
            this.page_result.UseVisualStyleBackColor = true;
            // 
            // lbl_correctness
            // 
            this.lbl_correctness.AutoSize = true;
            this.lbl_correctness.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.lbl_correctness.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lbl_correctness.ForeColor = System.Drawing.Color.Crimson;
            this.lbl_correctness.Location = new System.Drawing.Point(4, 351);
            this.lbl_correctness.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.lbl_correctness.Name = "lbl_correctness";
            this.lbl_correctness.Size = new System.Drawing.Size(98, 19);
            this.lbl_correctness.TabIndex = 12;
            this.lbl_correctness.Text = "Correctness : ";
            // 
            // btn_test
            // 
            this.btn_test.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btn_test.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btn_test.ForeColor = System.Drawing.Color.Indigo;
            this.btn_test.Location = new System.Drawing.Point(6, 293);
            this.btn_test.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.btn_test.Name = "btn_test";
            this.btn_test.Size = new System.Drawing.Size(74, 37);
            this.btn_test.TabIndex = 6;
            this.btn_test.Text = "Test";
            this.btn_test.UseVisualStyleBackColor = true;
            this.btn_test.Click += new System.EventHandler(this.btn_test_Click);
            // 
            // correctness_table
            // 
            this.correctness_table.AllowUserToAddRows = false;
            this.correctness_table.AllowUserToDeleteRows = false;
            this.correctness_table.AllowUserToResizeColumns = false;
            this.correctness_table.AllowUserToResizeRows = false;
            this.correctness_table.AutoSizeColumnsMode = System.Windows.Forms.DataGridViewAutoSizeColumnsMode.DisplayedCells;
            this.correctness_table.AutoSizeRowsMode = System.Windows.Forms.DataGridViewAutoSizeRowsMode.DisplayedCells;
            this.correctness_table.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.correctness_table.Dock = System.Windows.Forms.DockStyle.Top;
            this.correctness_table.Location = new System.Drawing.Point(2, 3);
            this.correctness_table.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.correctness_table.Name = "correctness_table";
            this.correctness_table.ReadOnly = true;
            this.correctness_table.RowTemplate.Height = 24;
            this.correctness_table.Size = new System.Drawing.Size(887, 284);
            this.correctness_table.TabIndex = 0;
            // 
            // progressBar
            // 
            this.progressBar.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.progressBar.Location = new System.Drawing.Point(0, 562);
            this.progressBar.Name = "progressBar";
            this.progressBar.Size = new System.Drawing.Size(899, 23);
            this.progressBar.TabIndex = 13;
            // 
            // OpenFile
            // 
            this.OpenFile.AccessibleName = "";
            this.OpenFile.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.OpenFile.Image = ((System.Drawing.Image)(resources.GetObject("OpenFile.Image")));
            this.OpenFile.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.OpenFile.Name = "OpenFile";
            this.OpenFile.Size = new System.Drawing.Size(52, 23);
            this.OpenFile.Text = "File";
            this.OpenFile.Click += new System.EventHandler(this.OpenFile_Click);
            // 
            // toolStrip
            // 
            this.toolStrip.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.OpenFile});
            this.toolStrip.Location = new System.Drawing.Point(0, 0);
            this.toolStrip.Name = "toolStrip";
            this.toolStrip.Size = new System.Drawing.Size(899, 26);
            this.toolStrip.TabIndex = 1;
            this.toolStrip.Text = "toolStrip1";
            // 
            // NeuralNetworkSystem
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(899, 584);
            this.Controls.Add(this.progressBar);
            this.Controls.Add(this.toolStrip);
            this.Controls.Add(this.tab);
            this.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.Name = "NeuralNetworkSystem";
            this.Text = "NeuralNetworkSystem";
            ((System.ComponentModel.ISupportInitialize)(this.chart)).EndInit();
            this.tab.ResumeLayout(false);
            this.page_main.ResumeLayout(false);
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel2.ResumeLayout(false);
            this.splitContainer1.Panel2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            this.splitContainer2.Panel1.ResumeLayout(false);
            this.splitContainer2.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer2)).EndInit();
            this.splitContainer2.ResumeLayout(false);
            this.splitContainer3.Panel1.ResumeLayout(false);
            this.splitContainer3.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer3)).EndInit();
            this.splitContainer3.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.num_perc)).EndInit();
            this.page_result.ResumeLayout(false);
            this.page_result.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.correctness_table)).EndInit();
            this.toolStrip.ResumeLayout(false);
            this.toolStrip.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.ListBox neuron_list;
		private System.Windows.Forms.Button btn_run;
		private System.Windows.Forms.DataVisualization.Charting.Chart chart;
		private System.Windows.Forms.PropertyGrid NN_system;
		private System.Windows.Forms.Label lbl_ein;
		private System.Windows.Forms.Label lbl_eout;
		private System.Windows.Forms.Button btn_del;
		private System.Windows.Forms.Label lbl_err01;
		private System.Windows.Forms.TabControl tab;
		private System.Windows.Forms.TabPage page_main;
		private System.Windows.Forms.TabPage page_result;
		private System.Windows.Forms.Button btn_test;
		private System.Windows.Forms.DataGridView correctness_table;
		private System.Windows.Forms.Label lbl_correctness;
		private System.Windows.Forms.Button btn_reset;
		private System.Windows.Forms.SplitContainer splitContainer1;
		private System.Windows.Forms.SplitContainer splitContainer2;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.NumericUpDown num_perc;
		private System.Windows.Forms.Label lbl_err01_best;
		private System.Windows.Forms.CheckBox stop_by_best;
		private System.Windows.Forms.Label lbl_errin01;
		private System.Windows.Forms.Button btn_ins;
		private System.Windows.Forms.ProgressBar progressBar;
		private System.Windows.Forms.ToolStripButton OpenFile;
		private System.Windows.Forms.ToolStrip toolStrip;
		private System.Windows.Forms.Panel panel;
		private System.Windows.Forms.SplitContainer splitContainer3;
	}
}

