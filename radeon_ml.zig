const std = @import("std");
const builtin = @import("builtin");

/// Maximal supported tensor rank
pub const tensor_max_rank = 5; // RML_TENSOR_MAX_RANK

/// Unspecified dimension value (a placeholder value)
pub const dim_unspecified = 0;
/// Device index for automatic device selection
pub const device_idx_unspecified = 0;

/// A storage for multiple strings
pub const Strings = extern struct {
    num_items: usize,
    items: [*c]const [*c]const u8,
};

// Returns a null-terminated string containing the last operation error message.
// May be called after some operation returns a status other than #RML_OK.
// The error message is owned by the library and must NOT be freed by a client.
// The message is stored in a thread local storage, so this function
// should be called from the thread where the failure occured.
//
// @return A pointer to the formatted message, in ASCII-encoding.
pub const getLastError = rmlGetLastError;
extern fn rmlGetLastError() [*:0]const u8;

// Set whether logging is enabled.
// @param[in] enabled Whether logging is enabled. Logging is enabled by default.
pub const setLoggingEnabled = @compileError("TODO: FIXME: Don't know why, but this function crashes at least on windows.");
// pub inline fn setLoggingEnabled(enabled: bool) void {
//     std.log.info("Enable logging: {}", .{@sizeOf(c_uint)});
//     rmlSetLoggingEnabled(@intFromBool(enabled));
//     std.log.info("done", .{});
// }
// extern fn rmlSetLoggingEnabled(enabled: c_uint) void;

/// A context handle
pub const Context = opaque {
    /// Context creation parameters.
    pub const Params = extern struct {
        // Device index, corresponding to the backend device query result.
        // Enumeration is started with 1. Use device_idx_unspecified (0)
        // for auto device selection.
        device_idx: u32 = device_idx_unspecified,
    };

    /// Creates a context.
    ///
    /// @param[in]  params  Context creation parameters, optional, @see #rml_context_params.
    /// @return A valid context handle in case of success or status:
    ///     - #RML_ERROR_BAD_PARAMETER if the @p params->device_idx is incorrect or the @p context is NULL,
    ///     - #RML_ERROR_INTERNAL in case of an internal error.
    /// To get more details in case of failure, call rmlGetLastError().
    /// The context should be released with rmlReleaseContext().
    pub inline fn initDefault(params: Params) Error!*Context {
        var ctx: *Context = undefined;
        try checkStatus(rmlCreateDefaultContext(params, &ctx));
        return ctx;
    }
    extern fn rmlCreateDefaultContext(params: Params, out_ctx: **Context) Status;

    /// Releases a context created with rmlCreateDefaultContext() or another context creating function,
    /// invalidates the handle.
    pub const deinit = rmlReleaseContext;
    extern fn rmlReleaseContext(ctx: *Context) void;
};

/// A tensor handle
pub const Tensor = opaque {
    // Creates an N-dimentional tensor with a given description.
    //
    // @param[in]  context A valid context handle.
    // @param[in]  info    Tensor description with all dimensions specified.
    // @param[in]  mode    Tensor data access mode.
    //
    // @return A valid tensor handle in case of success or status:
    // - #RML_OK if the operation is successful,
    // - #RML_ERROR_BAD_PARAMETER if @p context, @p info or @p mode is invalid or @p tensor is NULL,
    // - #RML_ERROR_OUT_OF_MEMORY if memory allocation is failed,
    // - #RML_ERROR_INTERNAL in case of an internal error.
    //
    // To get more details in case of failure, call rmlGetLastError().
    // The tensor should be released with rmlReleaseTensor().
    pub fn init(context: *Context, info: *const TensorInfo, mode: AccessMode) Error!Tensor {
        var tensor: *Tensor = undefined;
        try checkStatus(rmlCreateTensor(context, info, mode, &tensor));
        return tensor;
    }
    extern fn rmlCreateTensor(*Context, *const TensorInfo, AccessMode, **Tensor) Status;

    // Returns a tensor information.
    //
    // @param[in]  tensor A valid tensor handle.
    // @param[out] info   A pointer to a resulting info structure.
    //
    // @return Tensor information in case of success and status:
    // - #RML_OK if the operation is successful,
    // - #RML_ERROR_BAD_PARAMETER if @p tensor is invalid or @p info is NULL.
    //
    // To get more details in case of failure, call rmlGetLastError().
    ///
    pub fn getInfo(self: *Tensor) Error!TensorInfo {
        var res: *TensorInfo = undefined;
        try checkStatus(rmlGetTensorInfo(self, &res));
        return res;
    }
    extern fn rmlGetTensorInfo(*Tensor, *TensorInfo) Status;

    // Map the tensor data into the host address and returns a pointer to the mapped region.
    //
    // @param[in]  tensor A valid tensor handle.
    // @param[out] data   A pointer to a resulting data pointer.
    // @param[out] size   A pointer pointer to a resulting size. If not NULL, the referenced
    //                    size value is set to image size, in bytes.
    //
    // @return A data pointer, size (if @p size is not NULL) in case of success and status:
    // - #RML_OK if the operation is successful,
    // - #RML_ERROR_BAD_PARAMETER if @p tensor is invalid or @p data is NULL.
    //
    // To get more details in case of failure, call rmlGetLastError().
    // The mapped data must be unmapped with rmlUnmapTensor().
    pub fn map(self: *Tensor) []u8 {
        var data: *u8 = undefined;
        var size: usize = undefined;
        try checkStatus(rmlMapTensor(self, &data, &size));
        return data[0..size];
    }
    extern fn rmlMapTensor(*Tensor, data: **void, size: *usize) Status;

    // Unmaps a previously mapped tensor data.
    //
    // @param[in] tensor A valid tensor handle.
    // @param[in] data   A pointer to the previously mapped data.
    //
    // @return Status:
    // - #RML_OK if the operation is successful,
    // - #RML_ERROR_BAD_PARAMETER if @p data is invalid.
    //
    // To get more details in case of failure, call rmlGetLastError().
    pub fn unmap(self: *Tensor, data: []const u8) Error!void {
        try checkStatus(rmlUnmapTensor(self, data.ptr));
    }
    extern fn rmlUnmapTensor(*Tensor, data: *void) Status;

    // Releases an tensor created with rmlCreateTensor(), invalidates the handle.
    pub inline fn deinit(self: *Tensor) void {
        rmlReleaseTensor(self);
    }
    extern fn rmlReleaseTensor(*Tensor) void;
};

pub const Graph = opaque {
    /// Defines graph format required for loading model from buffer.
    pub const Format = enum(c_int) {
        /// Unspecified graph format.
        UNSPECIFIED = 0,
        /// Tensorflow 1.x binary graph format.
        TF = 400,
        /// Tensorflow text graph format.
        TF_TXT = 410,
        /// ONNX binary graph format.
        ONNX = 420,
        /// ONNX text graph format.
        ONNX_TXT = 430,
    };

    // Load graph from a ptotobuf file.
    //
    // @param[in]  path  Path to a graph in the TF or ONNX formats.
    // @param[out] graph The pointer to a resulting graph handle.
    //
    // @return A valid graph handle in case of success and status:
    // - #RML_OK if the operation is successful,
    // - #RML_ERROR_BAD_PARAMETER if @p path or @p graph is NULL,
    // - #RML_ERROR_FILE_NOT_FOUND if the model file is not found,
    // - #RML_ERROR_BAD_MODEL if the model contains an error.
    //
    // To get more details in case of failure, call rmlGetLastError().
    // The graph should be released with rmlReleaseGraph().
    pub fn loadFromFile(path: []const u8) Error!*Graph {
        var graph: *Graph = undefined;
        var wpath_buff : [std.fs.MAX_PATH_BYTES / @sizeOf(Char)] Char = undefined;
        try checkStatus(rmlLoadGraphFromFile(pathToWidePath(path, &wpath_buff), &graph));
        std.log.info("done", .{});
        return graph;
    }
    extern fn rmlLoadGraphFromFile(path: [*:0]const Char, out: **Graph) Status;

    // Loads graph from a protobuf buffer.
    //
    // @param[in]  size   The buffer size.
    // @param[in]  buffer The buffer pointer.
    // @param[in]  format The buffer format.
    // @param[out] graph  The pointer to a resulting graph handle.
    //
    // @return A valid graph handle in case of success and status:
    // - #RML_OK if the operation is successful,
    // - #RML_ERROR_BAD_PARAMETER if @p buffer or @p graph is NULL.
    // - #RML_ERROR_BAD_MODEL if the model contains an error.
    //
    // To get more details in case of failure, call rmlGetLastError().
    // The graph should be released with rmlReleaseGraph().
    pub inline fn loadFromBuffer(buffer: []const u8, format: Format) Error!Graph {
        var graph: *Graph = undefined;
        try checkStatus(rmlLoadGraphFromBuffer(buffer.len, buffer.ptr, format, &graph));
        return graph;
    }
    extern fn rmlLoadGraphFromBuffer(size: usize, buffer: *const anyopaque, format: Format, out: *Graph) Status;

    // Releases a graph created with rmlCreateGraph() or rmlLoadgraph(), invalidates the handle.
    pub const deinit = rmlReleaseGraph;
    extern fn rmlReleaseGraph(*Graph) void;
};

pub const Model = struct {
    // Creates a model from a supplied graph.
    //
    //  @param[in]  context A valid context handle.
    //  @param[in]  graph   A valid graph handle.
    //  @param[out] model   A pointer to a resulting model handle.
    //
    //  @return A model handle in case of success and status:
    //  - #RML_OK if the operation is successful,
    //  - #RML_ERROR_BAD_PARAMETER if @p context or @p graph is invalid or @p model is NULL.
    //
    //  To get more details in case of failure, call rmlGetLastError().
    pub inline fn initFromGraph(ctx: *Context, graph: *Graph) Error!*Model {
        var model: *Model = undefined;
        try checkStatus(rmlCreateModelFromGraph(ctx, graph, &model));
        return model;
    }
    extern fn rmlCreateModelFromGraph(*Context, *Graph, out: **Model) Status;

    // Releases a model created with rmlCreateModelFromGraph(), invalidates the handle.
    pub const deinit = rmlReleaseModel;
    extern fn rmlReleaseModel(*Model) void;

    // Sets up model output node names.
    //
    // If this function is not called, all leaf graph nodes are considered to be output.
    //
    // @param[in]  model A valid model handle.
    // @param[out] names A pointer to a structure with output names.
    //
    // @return Status:
    // - #RML_OK if the operation is successful,
    // - #RML_ERROR_BAD_PARAMETER if @p model or @p names is invalid.
    //
    // To get more details in case of failure, call rmlGetLastError().
    pub inline fn setOutputName(self: *Model, names: *const Strings) Error!void {
        try checkStatus(rmlSetModelOutputNames(self, names));
    }
    extern fn rmlSetModelOutputNames(*Model, names: *const Strings) Status;

    // Returns input tensor information by a node name.
    //
    // The @p name my be NULL if there is a single input (placeholder) node.
    //
    // @param[in]  model A valid model handle.
    // @param[in]  name  An optional input node name, in ASCII encoding.
    // @param[out] info  A pointer to a resulting input info structure.
    //                   If rmlSetModelInputInfo() was not previously called,
    //                   some dimensions may be unspecified.
    //
    // @return Input tensor information in case of success and status:
    // - #RML_OK if the operation is successful,
    // - #RML_ERROR_BAD_PARAMETER if @p model or @p name is invalid or @p info is NULL.
    //
    // To get more details in case of failure, call rmlGetLastError().
    pub inline fn getInputInfo(self: *Model, name: [*:0]const u8) Error!TensorInfo {
        var res: TensorInfo = undefined;
        try checkStatus(rmlGetModelInputInfo(self, name));
        return res;
    }
    extern fn rmlGetModelInputInfo(*Model, name: [*:0]const u8, out_info: *TensorInfo) Status;

    // Sets input tensor information for a node name.
    //
    // This call is optional if all model input dimensions are initially specified.
    // The @p name my be NULL if there is a single input (placeholder) node.
    //
    // @param[in] model A valid model handle.
    // @param[in] name  An input node name, in ASCII encoding.
    // @param[in] info  A pointer to a input info structure.
    //
    // @return Status:
    // - #RML_OK if the operation is successful,
    // - #RML_ERROR_BAD_PARAMETER if @p model or @p name is invalid or @p info is NULL.
    //
    // To get more details in case of failure, call rmlGetLastError().
    pub inline fn setInputInfo(self: *Model, name: [*:0]const u8, info: *const TensorInfo) !void {
        try checkStatus(rmlSetModelInputInfo(self, name, info));
    }
    extern fn rmlSetModelInputInfo(*Model, name: [*:0]const u8, info: *const TensorInfo) Status;

    // Returns output tensor information.
    //
    // All input dimensions must be specified before this call.
    // The @p name my be NULL if there is a single output node.
    //
    // @param[in]  model A valid model handle.
    // @param[in]  name  A optional output node name, in ASCII encoding.
    // @param[out] info  A pointer to a resulting output info structure.
    //
    // @return Output tensor information in case of success and status:
    // - #RML_OK if the operation is successful,
    // - #RML_ERROR_BAD_PARAMETER if @p model or @p name is invalid or @p info is NULL,
    // - #RML_ERROR_MODEL_NOT_READY if some inputs have unspecified dimensions.
    //
    // To get more details in case of failure, call rmlGetLastError().
    pub inline fn getOutputInfo(self: *Model, name: [*:0]const u8) Error!TensorInfo {
        var res: TensorInfo = undefined;
        try checkStatus(rmlGetModelOutputInfo(self, name, &res));
        return res;
    }
    extern fn rmlGetModelOutputInfo(*Model, name: [*:0]const u8, *TensorInfo) Status;

    // Returns memory usage information.
    //
    // All input dimensions must be specified before this call.
    //
    // @param[in]  model A valid model handle.
    // @param[out] info  A pointer to a resulting #rml_memory_info structure.
    //
    // @return Status:
    // - #RML_OK if the operation is successful,
    // - #RML_ERROR_BAD_PARAMETER if @p model is invalid or @p info is NULL,
    // - #RML_ERROR_MODEL_NOT_READY if some inputs have unspecified dimensions.
    pub inline fn getMemoryInfo(self: *Model) Status {
        var mem_info: MemoryInfo = undefined;
        try checkStatus(rmlGetModelMemoryInfo(self, &mem_info));
        return mem_info;
    }
    extern fn rmlGetModelMemoryInfo(*Model, *MemoryInfo) Status;

    // Sets up an input tensor for a node with a specified name.
    //
    // All input dimensions must be specified before this call.
    // The @p name my be NULL if there is a single input (placeholder) node.
    //
    // @param[in] model A valid model handle.
    // @param[in] name  A optional input node name, in ASCII encoding.
    // @param[in] input A valid input tensor handle.
    //
    // @return Status:
    // - #RML_OK if the operation is successful,
    // - #RML_ERROR_BAD_PARAMETER if @p model, @p name or @p input is invalid,
    // - #RML_ERROR_MODEL_NOT_READY if some inputs have unspecified dimensions.
    //
    // To get more details in case of failure, call rmlGetLastError().
    pub inline fn setInput(model: *Model, name: [*:0]const u8, input: Tensor) Error!void {
        try checkStatus(rmlSetModelInput(model, name, input));
    }
    extern fn rmlSetModelInput(model: *Model, name: [*:0]const u8, input: Tensor) Status;

    // Sets up an input tensor for a node with a specified name.
    //
    // All input dimensions must be specified before this call.
    // The @p name my be NULL if there is a single output node.
    //
    // @param[in] model  A valid model handle.
    // @param[in] name   An optional output node name, in ASCII encoding.
    // @param[in] output A valid output tensor handle.
    //
    // @return Status:
    // - #RML_OK if the operation is successful,
    // - #RML_ERROR_BAD_PARAMETER if @p model, @p name or @p output is invalid,
    // - #RML_ERROR_MODEL_NOT_READY if some inputs have unspecified dimensions.
    //
    // To get more details in case of failure, call rmlGetLastError().
    pub inline fn setOutput(self: *Model, name: [*:0]const u8, output: *Tensor) Error!void {
        try checkStatus(rmlSetModelOutput(self, name, output));
    }
    extern fn rmlSetModelOutput(*Model, name: [*:0]const u8, *Tensor) Status;

    // Prepares a model for inference.
    //
    // All model inputs must be set with rmlSetModelInput() and all model outputs
    // must be set with rmlSetModelOutput() before this function is called.
    //
    // @param[in] model A valid model handle.
    //
    // @return Status:
    // - #RML_OK if the operation is successful,
    // - #RML_ERROR_BAD_PARAMETER if @p model is invalid,
    // - #RML_ERROR_MODEL_NOT_READY if any input or output tensor is not specified,
    // - #RML_ERROR_OUT_OF_MEMORY if memory allocation is failed,
    // - #RML_ERROR_INTERNAL in case of an internal error.
    //
    // To get more details in case of failure, call rmlGetLastError().
    pub inline fn prepare(self: *Model) Error!void {
        try checkStatus(rmlPrepareModel(self));
    }
    extern fn rmlPrepareModel(*Model) Status;

    // Runs inference.
    //
    // All model inputs must be set with rmlSetModelInput() and all model outputs
    // must be set with rmlSetModelOutput() before this function is called.
    //
    // @param[in] model A valid model handle.
    //
    // @return Status:
    // - #RML_OK if the operation is successful,
    // - #RML_ERROR_BAD_PARAMETER if @p model is invalid,
    // - #RML_ERROR_MODEL_NOT_READY if any input or output tensor is not specified,
    // - #RML_ERROR_OUT_OF_MEMORY if memory allocation is failed,
    // - #RML_ERROR_INTERNAL in case of an internal error.
    //
    // To get more details in case of failure, call rmlGetLastError().
    pub inline fn infer(self: *Model) Error!void {
        try checkStatus(rmlInfer(self));
    }
    extern fn rmlInfer(*Model) Status;

    // Resets internal model states to their initial values.
    //
    // All model inputs must be set with rmlSetModelInput() and all model outputs
    // must be set with rmlSetModelOutput() before this function is called.
    //
    // @param[in] model A valid model handle.
    //
    // @return Status:
    // - #RML_OK if the operation is successful,
    // - #RML_ERROR_BAD_PARAMETER if @p model is invalid,
    // - #RML_ERROR_MODEL_NOT_READY if any input or output tensor is not specified,
    // - #RML_ERROR_INTERNAL in case of an internal error.
    //
    // To get more details in case of failure, call rmlGetLastError().
    pub inline fn resetStates(self: *Model) Error!void {
        try checkStatus(self);
    }
    extern fn rmlResetModelStates(*Model) Status;
};

// zig fmt: off
/// global error set that can be returned by functions, same as Status in C
pub const Error = error {
    BAD_MODEL,                // RML_ERROR_BAD_MODEL = -100,        // A model file has errors.
    BAD_PARAMETER,            // RML_ERROR_BAD_PARAMETER = -110,    // A parameter is incorrect.
    DEVICE_NOT_FOUND,         // RML_ERROR_DEVICE_NOT_FOUND = -120, // A device was not found.
    FILE_NOT_FOUND,           // RML_ERROR_FILE_NOT_FOUND = -130,   // A model file does not exist.
    INTERNAL,                 // RML_ERROR_INTERNAL = -140,         // An internal library error.
    MODEL_NOT_READY,          // RML_ERROR_MODEL_NOT_READY = -150,  // A model is not ready for an operation.
    NOT_IMPLEMENTED,          // RML_ERROR_NOT_IMPLEMENTED = -160,  // Functionality is not implemented yet.
    OUT_OF_MEMORY,            // RML_ERROR_OUT_OF_MEMORY = -170,    // Memory allocation is failed.
    UNSUPPORTED_DATA,         // RML_ERROR_UNSUPPORTED_DATA = -180, // An unsupported scenario.
};

/// Data type
pub const DType = enum(c_int) {
    UNSPECIFIED = 0,    //  Unspecified data type.
    FLOAT32 = 100,      //  Full precision float type.
    FLOAT16 = 101,      //  Half precision float type.
    UINT8 = 110,        //  Unsigned 8-bit integer type. Unsupported.
    INT32 = 120,        //  Signed 32-bit integer type.
};
// zig fmt: on

/// Physical memory layout of the tensor data.
pub const Layout = enum(c_int) {
    /// Unspecified layout
    UNSPECIFIED = 0,
    /// Tensor layout for a scalar value.
    SCALAR = 200,
    /// Tensor layout for a one dimensional tensor.
    C = 210,
    /// Tensor layout with the folowing dimensions: height, width
    HW = 220,
    /// Tensor layout for a two dimensional tensor with data stored in
    /// the row-major order, where C - number of elements in a column,
    /// N - number of elements in a row.
    NC = 221,
    /// Tensor layout for a single image in planar format
    /// with the following dimensions: number of channels, height, width.
    CHW = 230,
    /// Tensor layout for a single image in interleaved format
    /// with the following dimensions: height, width, number of channels.
    HWC = 231,
    /// Tensor layout with the following dimensions: number of images (batch
    /// size) , number of channels, height and width.
    NCHW = 240,
    /// Tensor layout with the following dimensions: number of images (batch
    /// size), height, width, number of channels.
    NHWC = 241,
    /// Tensor layout with the following dimensions: number of output
    /// channels, number of input channels, height, width.
    OIHW = 242,
    /// Tensor layout with the following dimensions: height, width, number
    /// of input channels, number of output channels.
    HWIO = 243,
};

/// Tensor access mode.
/// Mode indicates abilities to access tensor contents on a CPU.
pub const AccessMode = enum(c_int) {
    /// Unspecified access mode.
    UNSPECIFIED = 0,
    /// Allow reading from a tensor.
    READ_ONLY = 300,
    /// Allow reading from and writing to a tensor.
    READ_WRITE = 310,
    /// Allow writing from a tensor.
    WRITE_ONLY = 320,
    /// No reading from and writing to a tensor.
    DEVICE_ONLY = 330,
};

/// Memory information.
pub const MemoryInfo = extern struct {
    /// Total amount of allocated GPU memory.
    gpu_total: usize,
};

/// N-dimensional tensor description.
pub const TensorInfo = extern struct {
    /// Tensor data type.
    dtype: DType,
    /// Physical tensor data layout.
    layout: Layout,
    /// Tensor shape where axes order must correspond to the data layout.
    shape: u32[tensor_max_rank],
};

//
//  internal / private / hidden utils
//

const Status = enum(c_int) {
    RML_OK = 0, // Operation is successful.
    RML_ERROR_BAD_MODEL = -100, // A model file has errors.
    RML_ERROR_BAD_PARAMETER = -110, // A parameter is incorrect.
    RML_ERROR_DEVICE_NOT_FOUND = -120, // A device was not found.
    RML_ERROR_FILE_NOT_FOUND = -130, // A model file does not exist.
    RML_ERROR_INTERNAL = -140, // An internal library error.
    RML_ERROR_MODEL_NOT_READY = -150, // A model is not ready for an operation.
    RML_ERROR_NOT_IMPLEMENTED = -160, // Functionality is not implemented yet.
    RML_ERROR_OUT_OF_MEMORY = -170, // Memory allocation is failed.
    RML_ERROR_UNSUPPORTED_DATA = -180, // An unsupported scenario.
};

fn statusAsError(status: Status) Error {
    switch (status) {
        .RML_OK => unreachable,
        .RML_ERROR_BAD_MODEL => return error.BAD_MODEL,
        .RML_ERROR_BAD_PARAMETER => return error.BAD_PARAMETER,
        .RML_ERROR_DEVICE_NOT_FOUND => return error.DEVICE_NOT_FOUND,
        .RML_ERROR_FILE_NOT_FOUND => return error.FILE_NOT_FOUND,
        .RML_ERROR_INTERNAL => return error.INTERNAL,
        .RML_ERROR_MODEL_NOT_READY => return error.MODEL_NOT_READY,
        .RML_ERROR_NOT_IMPLEMENTED => return error.NOT_IMPLEMENTED,
        .RML_ERROR_OUT_OF_MEMORY => return error.OUT_OF_MEMORY,
        .RML_ERROR_UNSUPPORTED_DATA => return error.UNSUPPORTED_DATA,
    }
}

fn checkStatus(status: Status) Error!void {
    if (status != .RML_OK) return statusAsError(status);
}

/// An Unicode character
/// UTF-8 encoding is used for Linux and MacOS and UTF-16 is used for Windows
const Char = if (builtin.os.tag == .windows) std.os.windows.WCHAR else u8;
/// Create path that is null sentinel terminated, as well as convert from u8 to u16 if needed 
/// out_buff must be large enough to fit str + 1 (for 0 sentinel terminator)
fn pathToWidePath(str : []const u8, out_buff : []Char) [:0]Char {
    std.debug.assert(str.len + 1 < out_buff.len);
    var i : u32 = 0;
    while (i < str.len) : (i+=1) out_buff[i] = @intCast(str[i]);
    out_buff[i] = 0;
    return @ptrCast(out_buff[0..i]);
}
