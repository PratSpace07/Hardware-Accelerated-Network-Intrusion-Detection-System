`timescale 1ns / 1ps

module tb_pca_ids;

// ---------------------------------------------------------------------------
// Parameters â€” must match pca_anomaly_detector.v and your Python output
// ---------------------------------------------------------------------------
localparam N_FEAT    = 12;
localparam DATA_W    = 16;
localparam ACC_W     = 32;
localparam N_VEC     = 22544;   // total vectors in stimulus.mem
                                 // = number of rows in KDDTest+.csv after cleaning
                                 // Update this if your stimulus.mem has a different count
                                 // (Phase 1 Cell 9 prints the exact number)

// ---------------------------------------------------------------------------
// Clock and reset
// ---------------------------------------------------------------------------
reg clk = 0;
reg rst = 1;
always #5 clk = ~clk;   // 100 MHz clock, 10 ns period

// ---------------------------------------------------------------------------
// DUT signals
// ---------------------------------------------------------------------------
reg                          valid_in;
reg  [N_FEAT*DATA_W-1:0]     feat_vec;
wire                         valid_out;
wire signed [ACC_W-1:0]      score;
wire                         anomaly_flag;

// ---------------------------------------------------------------------------
// Instantiate DUT
// ---------------------------------------------------------------------------
pca_anomaly_detector #(
    .N_FEAT    (N_FEAT),
    .N_COMP    (4),
    .DATA_W    (DATA_W),
    .ACC_W     (ACC_W),
    .FRAC_BITS (12)
) dut (
    .clk         (clk),
    .rst         (rst),
    .valid_in    (valid_in),
    .feat_vec    (feat_vec),
    .valid_out   (valid_out),
    .score       (score),
    .anomaly_flag(anomaly_flag)
);

// ---------------------------------------------------------------------------
// Memory arrays
// stimulus_mem : flat array of all Q4.12 feature values
//   Layout: vector 0 features [0..11], vector 1 features [0..11], ...
//   Total entries = N_VEC * N_FEAT = 22544 * 12 = 270528
// labels_mem   : ground-truth binary (0=benign, 1=attack)
//   Total entries = N_VEC = 22544
// ---------------------------------------------------------------------------
reg [DATA_W-1:0] stimulus_mem [0:N_VEC*N_FEAT-1];
reg [0:0]        labels_mem   [0:N_VEC-1];

// ---------------------------------------------------------------------------
// Metrics counters
// ---------------------------------------------------------------------------
integer TP, FP, TN, FN;
integer vec_idx, feat_idx;
integer log_file;

// ---------------------------------------------------------------------------
// Main testbench body
// ---------------------------------------------------------------------------

   

initial begin
    // -- Load .mem files ------------------------------------------------------
    $readmemh("stimulus.mem", stimulus_mem);
    $readmemh("labels.mem",   labels_mem);

     #1;
    $display("STIM CHECK: stimulus_mem[0] = %h", stimulus_mem[0]);
    $display("STIM CHECK: stimulus_mem[1] = %h", stimulus_mem[1]);
    $display("LABEL CHECK: labels_mem[0] = %d", labels_mem[0]);

    // -- Open log file for offline Python scoring (optional) -----------------
    log_file = $fopen("sim_results.csv", "w");
    $fwrite(log_file, "vec_idx,flag,score,ground_truth\n");

    // -- VCD dump for waveform viewer ----------------------------------------
    $dumpfile("pca_ids_sim.vcd");
    $dumpvars(0, tb_pca_ids);

    // -- Initialise signals ---------------------------------------------------
    valid_in  = 0;
    feat_vec  = 0;
    TP = 0; FP = 0; TN = 0; FN = 0;

    // -- Reset sequence -------------------------------------------------------
    rst = 1;
    repeat(4) @(posedge clk);
    @(negedge clk);
    rst = 0;

    $display("=============================================================");
    $display("  PCA IDS Simulation Start");
    $display("  Vectors: %0d    Features: %0d", N_VEC, N_FEAT);
    $display("=============================================================");

    // -- Feed every vector into the DUT --------------------------------------
    for (vec_idx = 0; vec_idx < N_VEC; vec_idx = vec_idx + 1) begin

        // Pack N_FEAT features into the wide bus (LSB = feature 0)
        feat_vec = 0;
        for (feat_idx = 0; feat_idx < N_FEAT; feat_idx = feat_idx + 1)
            feat_vec[feat_idx*DATA_W +: DATA_W] =
                stimulus_mem[vec_idx*N_FEAT + feat_idx];

        // Assert valid_in for one clock cycle
        @(negedge clk);
        valid_in = 1;
        @(negedge clk);
        valid_in = 0;

        // Wait for DUT to finish (valid_out pulse)
        // With 6-state FSM and 1 cycle per state, this takes exactly 6 cycles
        // but we wait for the signal to be safe against future changes
        @(posedge valid_out);

        // -- Score this vector ------------------------------------------------
        if (labels_mem[vec_idx] == 1'b1) begin
            // Ground truth = ATTACK
            if (anomaly_flag) TP = TP + 1;
            else              FN = FN + 1;
        end else begin
            // Ground truth = BENIGN
            if (anomaly_flag) FP = FP + 1;
            else              TN = TN + 1;
        end

        // Write to CSV log
        $fwrite(log_file, "%0d,%0d,%0d,%0d\n",
                vec_idx, anomaly_flag, score, labels_mem[vec_idx]);

        // Print every 500th vector so you can watch progress
        if (vec_idx % 500 == 0)
            $display("  [%0t ns]  vec=%0d  flag=%b  score=%0d  gt=%0d",
                     $time, vec_idx, anomaly_flag, score, labels_mem[vec_idx]);
    end

    // -- Close log -----------------------------------------------------------
    $fclose(log_file);

    // -- Print final summary -------------------------------------------------
    $display("");
    $display("=============================================================");
    $display("  SIMULATION RESULTS");
    $display("=============================================================");
    $display("  Total vectors : %0d", N_VEC);
    $display("  Attack (GT=1) : %0d", TP + FN);
    $display("  Benign (GT=0) : %0d", FP + TN);
    $display("-------------------------------------------------------------");
    $display("  True  Positives (TP) : %0d", TP);
    $display("  False Positives (FP) : %0d", FP);
    $display("  True  Negatives (TN) : %0d", TN);
    $display("  False Negatives (FN) : %0d", FN);
    $display("-------------------------------------------------------------");

    // Detection Rate and False Alarm Rate (integer percentage for $display)
    if ((TP + FN) > 0)
        $display("  Detection Rate (DR)      : %0d.%02d%%",
                 (TP * 100) / (TP + FN),
                 ((TP * 10000) / (TP + FN)) % 100);
    else
        $display("  Detection Rate (DR)      : N/A (no attack samples)");

    if ((FP + TN) > 0)
        $display("  False Alarm Rate (FAR)   : %0d.%02d%%",
                 (FP * 100) / (FP + TN),
                 ((FP * 10000) / (FP + TN)) % 100);
    else
        $display("  False Alarm Rate (FAR)   : N/A (no benign samples)");

    $display("=============================================================");
    $display("  Compare these numbers against Python Phase 2 output.");
    $display("  DR match within ~1%% confirms fixed-point implementation");
    $display("  is mathematically equivalent to the Python golden model.");
    $display("=============================================================");

    // Also print to sim_results.csv path reminder
    $display("  Detailed results written to: sim_results.csv");
    $display("  VCD waveform written to:     pca_ids_sim.vcd");

    #100;
    $finish;
end

// ---------------------------------------------------------------------------
// Timeout watchdog â€” prevents infinite hang if valid_out never arrives
// Adjust if N_VEC is large; 10 cycles * N_VEC * 2 is a safe upper bound
// ---------------------------------------------------------------------------
initial begin
    #(N_VEC * 10 * 20);   // N_VEC vectors * 10 cycles each * 10 ns/cycle * 2 margin
    $display("ERROR: Simulation timeout â€” DUT may be hung. Check FSM transitions.");
    $finish;
end

endmodule
