#ifndef __ASYNC_LOG_HPP__
#define __ASYNC_LOG_HPP__

#include <any>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <variant>

#include "fmt/format.h"

#define ASYNC_LOG_DEBUG(...)                                                  \
  AsyncLog::AsyncLog::instance().AsyncWrite(AsyncLog::LogLv::DEBUG, __FILE__, \
                                            __LINE__, __VA_ARGS__);

#define ASYNC_LOG_INFO(...)                                                  \
  AsyncLog::AsyncLog::instance().AsyncWrite(AsyncLog::LogLv::INFO, __FILE__, \
                                            __LINE__, __VA_ARGS__);

#define ASYNC_LOG_WARN(...)                                                  \
  AsyncLog::AsyncLog::instance().AsyncWrite(AsyncLog::LogLv::WARN, __FILE__, \
                                            __LINE__, __VA_ARGS__);

#define ASYNC_LOG_ERROR(...)                                                  \
  AsyncLog::AsyncLog::instance().AsyncWrite(AsyncLog::LogLv::ERROR, __FILE__, \
                                            __LINE__, __VA_ARGS__);

#define ASYNC_LOG_FATAL(...)                                                  \
  AsyncLog::AsyncLog::instance().AsyncWrite(AsyncLog::LogLv::FATAL, __FILE__, \
                                            __LINE__, __VA_ARGS__);

namespace AsyncLog {

static const char *const kRst = "\033[0m";

static const char *const kBlack = "\033[30m";
static const char *const kRed = "\033[31m";
static const char *const kGreen = "\033[32m";
static const char *const kYellow = "\033[33m";
static const char *const kBlue = "\033[34m";
static const char *const kMagenta = "\033[35m";
static const char *const kCyan = "\033[36m";
static const char *const kWhite = "\033[37m";

static const char *const kBlackBr = "\033[1;30m";
static const char *const kRedBr = "\033[1;31m";
static const char *const kGreenBr = "\033[1;32m";
static const char *const kYellowBr = "\033[1;33m";
static const char *const kBlueBr = "\033[1;34m";
static const char *const kMagentaBr = "\033[1;35m";
static const char *const kCyanBr = "\033[1;36m";
static const char *const kWhiteBr = "\033[1;37m";

static const char *const kBlackBg = "\033[40m";
static const char *const kRedBg = "\033[41m";
static const char *const kGreenBg = "\033[42m";
static const char *const kYellowBg = "\033[43m";
static const char *const kBlueBg = "\033[44m";
static const char *const kMagentaBg = "\033[45m";
static const char *const kCyanBg = "\033[46m";
static const char *const kWhiteBg = "\033[47m";

enum LogLv { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3, FATAL = 4 };

static const std::array<const char *, LogLv::FATAL + 1> level_msgs = {
    "DEBUG", "INFO", "WARN", "ERROR", "FATAL"};

static const std::array<const char *, LogLv::FATAL + 1> level_colors = {
    kBlue, kGreen, kYellow, kRed, kRedBg};

class LogTaskBase {
 public:
  LogTaskBase(LogLv level) : _level(level) {
    _now = std::chrono::system_clock::now().time_since_epoch().count();
  }
  virtual ~LogTaskBase() = default;
  virtual void GetString(std::string &log_prifix, std::string &log_info) = 0;
  LogLv _level;
  int64_t _now;
};

template <typename... Args>
class LogTask : public LogTaskBase {
 public:
  template <typename... TArgs>
  LogTask(LogLv level, const char *file, int line, std::string format_data,
          TArgs &&... log_data)
      : LogTaskBase(level),
        _file_name(file),
        _line(line),
        _format_data(format_data),
        _log_data(std::forward<TArgs>(log_data)...) {}

  // Optionally, you can provide getter functions for the level and log_data.
  void GetString(std::string &log_prifix, std::string &log_info) {
    log_info = std::apply(
        [&](auto &&... args) {
          return fmt::format(_format_data,
                             args...);  // Formatting the log data
        },
        _log_data);

    size_t lastSlash = _file_name.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
      _file_name = _file_name.substr(lastSlash + 1);
    }

    log_prifix = std::move(
        std::string("[" + _file_name + ":" + std::to_string(_line) + "] "));
  }

 private:
  std::string _format_data;
  std::string _file_name;
  int _line;
  std::tuple<Args...> _log_data;
};

class AsyncLog {
 public:
  static AsyncLog &instance() {
    static AsyncLog instance;
    return instance;
  }

  ~AsyncLog() {
    Stop();
    _work_thread->join();
  }

  void Stop() {
    _stop.store(true, std::memory_order_relaxed);
    _empty_cond.notify_one();
  }

  template <typename... Args>
  void AsyncWrite(LogLv level, const char *file, int line,
                  const std::string &format_data, Args &&... args) {
    std::unique_ptr<LogTaskBase> task = std::make_unique<LogTask<Args...>>(
        level, file, line, format_data, std::forward<Args>(args)...);
    bool notify = false;
    {
      std::unique_lock<std::mutex> lock(_log_task_mtx);
      _log_tasks.push(std::move(task));
      notify = _log_tasks.size() == 1;
    }
    if (notify) {
      _empty_cond.notify_one();
    }
  }

  void SetLevel(LogLv level) { _level_limit = level; }

 private:
  AsyncLog() : _stop(false), _work_thread{nullptr}, _level_limit(LogLv::INFO) {
    _work_thread = std::make_unique<std::thread>([this]() {
      for (;;) {
        std::unique_lock<std::mutex> lock(_log_task_mtx);
        _empty_cond.wait(lock, [this]() {
          return _stop.load(std::memory_order_acq_rel) || !_log_tasks.empty();
        });

        if (_stop.load(std::memory_order_acq_rel)) {
          break;
        }

        std::unique_ptr<LogTaskBase> log_task = std::move(_log_tasks.front());
        _log_tasks.pop();
        lock.unlock();

        LogLv level = log_task->_level;
        if (level < _level_limit) {
          continue;
        }
        std::string log_prifix;
        std::string log_info;
        log_task->GetString(log_prifix, log_info);
        printf("[%s] %s %s %s %s %s\n",
               FormatNanoTimestamp(log_task->_now).c_str(), level_colors[level],
               log_prifix.c_str(), level_msgs[level], log_info.c_str(), kRst);
      }
      std::cout << "exit log work thread..." << std::endl;
    });
  }

  static std::string FormatNanoTimestamp(int64_t nano) {
    auto ms = nano / 1000000;
    // 使用 std::put_time 将 std::chrono::milliseconds
    // 转换为格式化的日期字符串（精度为毫秒）
    std::time_t time =
        static_cast<std::time_t>(ms / 1000);  // Convert milliseconds to seconds
    char buffer[64];  // Buffer to hold the formatted date
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S",
                  std::localtime(&time));

    // Append milliseconds to the formatted date
    std::stringstream ss;
    ss << buffer << "." << std::setw(3) << std::setfill('0') << ms % 1000;
    return ss.str();
  }

  AsyncLog(const AsyncLog &) = delete;
  AsyncLog &operator=(const AsyncLog &) = delete;

 private:
  std::atomic_bool _stop;
  std::unique_ptr<std::thread> _work_thread;
  std::queue<std::unique_ptr<LogTaskBase>> _log_tasks;
  std::mutex _log_task_mtx;
  std::condition_variable _empty_cond;
  LogLv _level_limit;
};

}  // namespace AsyncLog

#endif