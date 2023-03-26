#ifndef BMF_SUITE_MENU_H
#define BMF_SUITE_MENU_H

#include <ncurses.h>
#include <set>
#include <map>
#include <stack>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <functional>

enum MenuNavMode {
    NAV_NORMAL,
    NAV_PEND_TERMINATE,
    NAV_PEND_CONFIRM
};

enum MenuType {
    MENU_SINGLE,
    MENU_MULTI,
    MENU_OPTION,
    MENU_CONFIRM,
    MENU_BACK,
    MENU_QUIT
};

enum MenuOptionTag {
    TAG_NORMAL,
    TAG_SELECT_ALL
};

class Menu {
public:
    std::string title;
    std::vector<std::string> pretext;
    std::vector<Menu> options;
    std::set<int> selections;
    MenuType menu_type = MENU_SINGLE;
    MenuOptionTag option_tag = TAG_NORMAL;
    int cursor = 0;
    std::function<void()> callback = [this]() {};
    Menu (std::string text) : title(text) {}
    void clear_screen();
    void update_screen();
    void display();
    void next();
    void prev();
    bool select_option();
    Menu& Text(std::string text);
    Menu& SubMenu(Menu submenu);
    Menu& Option(std::string text, MenuOptionTag tag = TAG_NORMAL);
    Menu& Back(std::string text = "返回上一页");
    Menu CreateQuitMenu();
    Menu& Quit();
    Menu& Confirmation();
    Menu& get_selected_menu();
    Menu& Acceptance(std::function<void()> cb);
};

static std::map<std::string, std::string> config;
static std::set<int> selected_options;

#endif //BMF_SUITE_MENU_H