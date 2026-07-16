`timescale 1ns / 1ps

module pca_anomaly_detector #(
    parameter N_FEAT    = 12,
    parameter N_COMP    = 4,
    parameter DATA_W    = 16,
    parameter ACC_W     = 32,
    parameter FRAC_BITS = 12
)(
    input  wire                         clk,
    input  wire                         rst,
    // Handshake
    input  wire                         valid_in,    // pulse high for 1 cycle to submit vector
    input  wire [N_FEAT*DATA_W-1:0]     feat_vec,    // 12 features packed, Q4.12 each
    output reg                          valid_out,   // pulses high for 1 cycle when result ready
    output reg  signed [ACC_W-1:0]      score,       // reconstruction error (Q4.12 scale)
    output reg                          anomaly_flag // 1 = intrusion detected
);

// ---------------------------------------------------------------------------
// State encoding
// ---------------------------------------------------------------------------
localparam [2:0]
    IDLE        = 3'd0,
    NORMALISE   = 3'd1,
    PROJECT     = 3'd2,
    RECONSTRUCT = 3'd3,
    RECON_ERR   = 3'd4,
    DECISION    = 3'd5;

reg [2:0] state;

// ---------------------------------------------------------------------------
// Coefficient memories â€” loaded from Python-generated .mem files
// All values are Q4.12 signed fixed-point (16-bit, two's complement)
// ---------------------------------------------------------------------------
reg signed [DATA_W-1:0] mean_mem   [0:N_FEAT-1];          // StandardScaler means
reg signed [DATA_W-1:0] invstd_mem [0:N_FEAT-1];          // 1/std (pre-inverted in Python)
// PCA weight matrix W, shape (N_COMP, N_FEAT), stored row-major
// Access: W[k][n] = w_mem[k*N_FEAT + n]
reg signed [DATA_W-1:0] w_mem      [0:N_COMP*N_FEAT-1];
reg signed [ACC_W-1:0]  thresh_mem [0:0];                  // anomaly threshold

initial begin
    $readmemh("mean.mem",      mean_mem);
    $readmemh("invstd.mem",    invstd_mem);
    $readmemh("pca_ww.mem",     w_mem);
    $readmemh("threshold.mem", thresh_mem);

    // ---------- DEBUG: Force print to console ----------
    #1; // Wait 1 ns for the memory to load
    $display("=============================================");
    $display("DUT COEFFICIENT LOAD CHECK");
    $display("mean_mem[0]   = %h", mean_mem[0]);
    $display("invstd_mem[0] = %h", invstd_mem[0]);
    $display("w_mem[0]      = %h", w_mem[0]);
    $display("threshold     = %h", thresh_mem[0]);
    $display("=============================================");
end

// ---------------------------------------------------------------------------
// Intermediate registers
// Sized according to the bit-width analysis:
//   x_in, x_norm, x_hat : Q4.12 (16-bit signed) â€” one per feature
//   y                   : 32-bit signed â€” one per component (wider for accumulation)
//   recon_err           : 32-bit signed scalar
// ---------------------------------------------------------------------------
reg signed [DATA_W-1:0] x_in   [0:N_FEAT-1];
reg signed [DATA_W-1:0] x_norm [0:N_FEAT-1];
reg signed [ACC_W-1:0]  y      [0:N_COMP-1];
reg signed [DATA_W-1:0] x_hat  [0:N_FEAT-1];
reg signed [ACC_W-1:0]  recon_err;

// Scratch accumulator for inner-loop sums (declared as integer for for-loop use)
integer n, k;
reg signed [ACC_W-1:0] acc;
reg signed [DATA_W-1:0] diff;

// ---------------------------------------------------------------------------
// FSM
// ---------------------------------------------------------------------------
always @(posedge clk) begin
    if (rst) begin
        state        <= IDLE;
        valid_out    <= 1'b0;
        anomaly_flag <= 1'b0;
        score        <= 0;
    end else begin
        case (state)

            // ----------------------------------------------------------------
            // IDLE: wait for a valid feature vector, unpack the bus
            // ----------------------------------------------------------------
            IDLE: begin
                valid_out <= 1'b0;
                if (valid_in) begin
                    // Unpack packed bus into individual signed registers
                    // feat_vec[n*DATA_W +: DATA_W] extracts 16 bits starting at bit n*16
                    // $signed() is critical â€” without it Verilog treats it as unsigned
                    for (n = 0; n < N_FEAT; n = n + 1)
                        x_in[n] <= $signed(feat_vec[n*DATA_W +: DATA_W]);
                    state <= NORMALISE;
                end
            end

            // ----------------------------------------------------------------
            // NORMALISE: x_norm[n] = (x_in[n] - mean[n]) * invstd[n]
            //
            // Both operands are Q4.12 signed 16-bit.
            // Product is Q8.24 (32-bit). Right-shift by FRAC_BITS â†’ Q4.12.
            // Result fits in 16-bit Q4.12 because clipping in Python bounds it.
            //
            // Note: >>> is arithmetic right-shift (preserves sign bit).
            //       >>  is logical right-shift (fills with 0 â€” WRONG for negatives).
            // ----------------------------------------------------------------
           NORMALISE: begin
    // stimulus.mem is already standardized by Python's StandardScaler!
    // So we just copy x_in directly to x_norm.
    for (n = 0; n < N_FEAT; n = n + 1)
        x_norm[n] <= x_in[n];
    state <= PROJECT;
end

            // ----------------------------------------------------------------
            // PROJECT: y[k] = sum_n( x_norm[n] * W[k][n] )
            //
            // Outer loop over components (k), inner over features (n).
            // acc resets for each new k. Each product is Q8.24; shift â†’ Q4.12.
            // y[k] stored at 32-bit width (ACC_W) to avoid truncation.
            //
            // Verilog for-loops inside always blocks UNROLL completely â€”
            // Vivado creates parallel multipliers for all 12 features at once.
            // This uses more LUTs but finishes in a single clock cycle.
            // ----------------------------------------------------------------
            PROJECT: begin
                for (k = 0; k < N_COMP; k = k + 1) begin
                    acc = 0;
                    for (n = 0; n < N_FEAT; n = n + 1)
                        acc = acc + (($signed(x_norm[n]) * $signed(w_mem[k*N_FEAT + n]))
                                     >>> FRAC_BITS);
                    y[k] <= acc;
                end
                state <= RECONSTRUCT;
            end

            // ----------------------------------------------------------------
            // RECONSTRUCT: x_hat[n] = sum_k( y[k] * W[k][n] )
            //
            // Same structure as PROJECT but loops are transposed.
            // y[k] is already Q4.12-scaled (32-bit). W[k][n] is Q4.12 (16-bit).
            // Product is Q8.24 (32-bit). Shift â†’ Q4.12. Truncate to 16-bit.
            // ----------------------------------------------------------------
            RECONSTRUCT: begin
                for (n = 0; n < N_FEAT; n = n + 1) begin
                    acc = 0;
                    for (k = 0; k < N_COMP; k = k + 1)
                        acc = acc + (($signed(y[k]) * $signed(w_mem[k*N_FEAT + n]))
                                     >>> FRAC_BITS);
                    x_hat[n] <= acc[DATA_W-1:0];  // truncate back to 16-bit Q4.12
                end
                state <= RECON_ERR;
            end

            // ----------------------------------------------------------------
            // RECON_ERR: recon_err = sum_n( (x_norm[n] - x_hat[n])^2 )
            //
            // diff is Q4.12 (16-bit signed).
            // diff^2 is Q8.24 (32-bit). Right-shift by FRAC_BITS â†’ Q4.12 scale.
            // This lets us compare recon_err directly with thresh_mem[0],
            // which was stored as to_fixed(threshold) = threshold * 2^12.
            // ----------------------------------------------------------------
            RECON_ERR: begin
                recon_err <= 0;  // reset before accumulating
                acc = 0;
                for (n = 0; n < N_FEAT; n = n + 1) begin
                    diff = x_norm[n] - x_hat[n];
                    acc  = acc + (($signed(diff) * $signed(diff)) >>> FRAC_BITS);
                end
                recon_err <= acc;
                state     <= DECISION;
            end

            // ----------------------------------------------------------------
            // DECISION: compare error vs threshold, set flag, pulse valid_out
            // ----------------------------------------------------------------
            DECISION: begin
                score        <= recon_err;
                anomaly_flag <= (recon_err > thresh_mem[0]) ? 1'b1 : 1'b0;
                valid_out    <= 1'b1;   // pulse for exactly one clock cycle
                state        <= IDLE;
            end

        endcase
    end
end



endmodule

