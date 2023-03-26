
#include <bmf/sdk/common.h>
#include <bmf/sdk/trace.h>

USE_BMF_SDK_NS

int main(int argc, char ** argv) {
#ifndef NO_TRACE
    // Perform log formatting to tracelog without including trace tool's
    // additional information e.g. on buffer capacity etc
    TraceLogger::instance()->format_logs(false);
#endif
    return 0;
}