
#include <stdio.h>
#define GL_GLEXT_PROTOTYPES
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <GLES3/gl3.h>
#include <EGL/eglext.h>
namespace egl {

class EGLHelper {
  public:
    bool m_initilized = false;
    EGLDisplay m_eglDisplay = EGL_NO_DISPLAY;
    EGLSurface m_eglSurface = EGL_NO_SURFACE;
    EGLContext m_eglContext;

    EGLint m_eglMajorVersion = 0;
    EGLint m_eglMinorVersion = 0;

  public:
    EGLHelper(int scr_width, int scr_height) {
        // 1. Initialize EGL
        m_eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (m_eglDisplay == EGL_NO_DISPLAY) {
            printf("eglGetDisplay returned EGL_NO_DISPLAY.\n");
            return;
        }

        m_eglMajorVersion = 0;
        m_eglMinorVersion = 0;
        EGLBoolean rc =
            eglInitialize(m_eglDisplay, &m_eglMajorVersion, &m_eglMinorVersion);
        if (rc != EGL_TRUE) {
            printf("eglInitialize failed\n");
            return;
        }

        // 2. Select an appropriate configuration
        EGLint numConfigs;
        EGLConfig eglCfg;
        static const EGLint configAttribs[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RED_SIZE,     8,
            EGL_GREEN_SIZE,   8,
            EGL_BLUE_SIZE,    8,
            EGL_ALPHA_SIZE,   8,
            EGL_DEPTH_SIZE,   8,
            EGL_NONE};
        if (!eglChooseConfig(m_eglDisplay, configAttribs, &eglCfg, 1,
                             &numConfigs)) {
            printf("eglChooseConfig error\n");
            return;
        }

        static const EGLint pbufferAttribs[] = {
            EGL_WIDTH,           scr_width, EGL_HEIGHT, scr_height,
            EGL_LARGEST_PBUFFER, EGL_TRUE,  EGL_NONE,
        };
        // 3. Create a offscreen surface
        m_eglSurface =
            eglCreatePbufferSurface(m_eglDisplay, eglCfg, pbufferAttribs);
        if (m_eglSurface == EGL_NO_SURFACE) {
            return;
        }
        // 4. set current render api to es
        eglBindAPI(EGL_OPENGL_ES_API);

        // 5. Create a context and make it current
        int m_glesMajorVersion = 3;
        int m_glesMinorVersion = 0;
        const EGLint contextAttribs[] = {
            EGL_CONTEXT_MAJOR_VERSION, static_cast<int>(m_glesMajorVersion),
            EGL_CONTEXT_MINOR_VERSION, static_cast<int>(m_glesMinorVersion),
            EGL_NONE};
        m_eglContext = eglCreateContext(m_eglDisplay, eglCfg, EGL_NO_CONTEXT,
                                        contextAttribs);

        // 6. make display and surface current
        eglMakeCurrent(m_eglDisplay, m_eglSurface, m_eglSurface, m_eglContext);

        m_initilized = true;
    }

    bool initilized() const { return m_initilized; }

    int create_oes_texture(int width, int height, EGLImageKHR &mEGLImage,
                           unsigned int &hold_texture) {
        unsigned int texture;
        glGenTextures(1, &texture);

        glGenTextures(1, &hold_texture);
        glBindTexture(GL_TEXTURE_2D, hold_texture);
        // set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        // set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // load image, create texture and generate mipmaps
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, 0);

        mEGLImage = eglCreateImageKHR(m_eglDisplay, eglGetCurrentContext(),
                                      EGL_GL_TEXTURE_2D,
                                      (EGLClientBuffer)hold_texture, 0);
        if (EGL_NO_IMAGE == mEGLImage) {
            return 0;
        }

        glBindTexture(GL_TEXTURE_EXTERNAL_OES, texture);
        glEGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES,
                                     (GLeglImageOES)mEGLImage);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER,
                        GL_LINEAR);

        glBindTexture(GL_TEXTURE_EXTERNAL_OES, 0);
        return texture;
    }
    int create_texture(int width, int height, bool texture_storage_2d = false) {
        if (!m_initilized) {
            return -1;
        }

        unsigned int texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        // set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        // set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // load image, create texture and generate mipmaps
        if (texture_storage_2d) {
            printf("use glTexStorage2D.\n");
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8UI, width, height);
        } else {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, 0);
        }
        return texture;
    }

    void release_texture(int texture) {
        glDeleteTextures(1, (unsigned int *)&texture);
    }
    void release_oes_texture(int texture, EGLImage mEGLImage,
                             int hold_texture) {
        glDeleteTextures(1, (unsigned int *)&texture);
        eglDestroyImageKHR(eglGetCurrentDisplay(), mEGLImage);
        glDeleteTextures(1, (unsigned int *)&texture);
    }
    void copy_to_texture(int texture, void *data, int width, int height,
                         bool texture_storage_2d = false) {
        glBindTexture(GL_TEXTURE_2D, texture);
        if (texture_storage_2d) {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                            GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, data);
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA,
                            GL_UNSIGNED_BYTE, data);
        }
    }

    void copy_from_texture(int texture, void *data, int width, int height,
                           bool texture_storage_2d = false) {
        glBindTexture(GL_TEXTURE_2D, texture);

        unsigned int fbo;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, texture, 0);
        if (texture_storage_2d) {
            glReadPixels(0, 0, width, height, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE,
                         data);
        } else {
            glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
    }
};

} // namespace egl
