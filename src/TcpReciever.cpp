#include "video/TcpReciever.h"
#include <arpa/inet.h>
#include <cstddef>          //  <-- new
#include <cstring>
#include <netinet/in.h>
#include <unistd.h>
#include <iostream>

namespace vs {

namespace {
constexpr int    BACKLOG = 4;
constexpr size_t BUF_SZ  = 256;
}

/* ------------------------------------------------------------------ */
TcpReciever::TcpReciever(uint16_t port) : port_(port) {}
TcpReciever::~TcpReciever() { stop(); }

/* ------------------------------------------------------------------ */
bool TcpReciever::start()
{
    if (running_) return true;

    listenFd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd_ < 0) return false;

    int on = 1;
    ::setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port_);

    if (::bind(listenFd_, (sockaddr*)&addr, sizeof(addr)) < 0 ||
        ::listen(listenFd_, BACKLOG)                       < 0)
    {
        ::close(listenFd_);
        listenFd_ = -1;
        return false;
    }

    running_ = true;
    thread_  = std::thread(&TcpReciever::listenLoop, this);
    return true;
}

/* ------------------------------------------------------------------ */
void TcpReciever::stop()
{
    if (!running_) return;
    running_ = false;

    ::shutdown(listenFd_, SHUT_RDWR);
    ::close(listenFd_);
    listenFd_ = -1;

    if (thread_.joinable()) thread_.join();
}

/* ------------------------------------------------------------------ */
bool TcpReciever::tryGetLatest(int& outX, int& outY)
{
    int x = latestX_.exchange(-1);
    int y = latestY_.exchange(-1);

    if (x < 0 || y < 0) return false;
    outX = x;  outY = y;
    return true;
}

/* ------------------------------------------------------------------ */
void TcpReciever::listenLoop()
{
    while (running_) {
        int client = ::accept(listenFd_, nullptr, nullptr);
        if (client < 0) {
            if (running_) continue;
            break;
        }

        char        buf[BUF_SZ];
        ssize_t     n;
        std::string pending;

        while (running_ && (n = ::recv(client, buf, sizeof(buf), 0)) > 0) {
            pending.append(buf, n);

            size_t pos;
            while ((pos = pending.find('\n')) != std::string::npos) {
                std::string line = pending.substr(0, pos);
                pending.erase(0, pos + 1);

                int x, y;
                if (std::sscanf(line.c_str(), "%d %d", &x, &y) == 2) {
                    latestX_ = x;
                    latestY_ = y;
                }
                std::cout << "got (" << x << ',' << y << ")\n";
            }
        }
        ::close(client);
    }
}

} // namespace vs
