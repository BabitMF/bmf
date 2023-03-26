//
//

#ifndef BMF_ENGINE_ENV_INIT_H
#define BMF_ENGINE_ENV_INIT_H

#include <string>
#ifdef BMF_ENABLE_BREAKPAD
#include "client/linux/handler/exception_handler.h"
#endif

namespace bmf::internal {
    class EnvInit {
    public:
#ifdef BMF_ENABLE_BREAKPAD
        google_breakpad::ExceptionHandler *handler;
#endif
        EnvInit();
        void ChangeDmpPath(std::string path);
    };

    inline EnvInit env_init;
}

#endif //BMF_ENGINE_ENV_INIT_H
