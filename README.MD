# zig-radeon-ml

Zig bindings to RadeonML: https://github.com/GPUOpen-LibrariesAndSDKs/RadeonML  
In comparison to raw headers:
* implements zig error handling
* some additional type safety, use of slices etc.
* functions grouped within as struct/opaque members
* zig naming convention, removed unnececary prefixes as in zig its already namespaced, minor things like create/destroy renamed to init&deinit


### Implementation status

Implemented, just needs cleaning doc comments (currently kept from c headers) and more testing.
* ✔️ [RadeonML.hpp](include/rml/RadeonML.h)

TODO: 
* graph manipulation and interoperation headers
* packagin with zig pkg manager and maybe utilities to help install *.dll   
