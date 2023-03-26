#include "../include/menu.h"

void Menu::clear_screen() {
    clear();
    cursor = 0;
    selections.clear();
}

void Menu::update_screen() {
    int next_row = 0;
    for (auto line : pretext) {
        mvprintw(next_row, 0, line.c_str());
        next_row++;
    }
    for (int i = 0; i < options.size(); i++) {
        std::string prefix = "    ";
        if (menu_type == MENU_MULTI && options[i].menu_type == MENU_OPTION) {
            if (cursor == i)
                prefix = " [>]";
            else if (selections.count(i))
                prefix = " [X]";
            else
                prefix = " [ ]";
        } else if (cursor == i)
            prefix = "  > ";
            
        mvprintw(next_row, 0, (prefix + options[i].title).c_str());
        next_row++;
    }
    refresh();
}

void Menu::display() {
    clear_screen();
    update_screen();
}

void Menu::next() {
    cursor++;
    if (cursor == options.size())
        cursor = 0;
    update_screen();
}

void Menu::prev() {
    cursor--;
    if (cursor < 0)
        cursor = options.size() - 1;
    update_screen();
}

bool Menu::select_option() {
    if (options[cursor].menu_type == MENU_OPTION) {
        if (options[cursor].option_tag == TAG_SELECT_ALL) {
            if (selections.count(cursor)) {
                selections.clear();
            } else {
                for (int i = 0; i < options.size(); i++) {
                    if (options[i].menu_type == MENU_OPTION)
                        selections.insert(i);
                }
            }
        } else {
            if (selections.count(cursor)) {
                selections.erase(cursor);
                for (int i = 0; i < options.size(); i++) {
                    if (options[i].option_tag == TAG_SELECT_ALL) {
                        selections.erase(i);
                    }
                }
            } else {
                selections.insert(cursor);
            }
        }
        update_screen();
        return true;
    }
    return false;
}

Menu& Menu::Text(std::string text) {
    pretext.push_back(text);
    return *this;
}

Menu& Menu::SubMenu(Menu submenu) {
    options.push_back(submenu);
    return *this;
}

Menu& Menu::Option(std::string text, MenuOptionTag tag) {
    Menu menu = Menu(text);
    menu_type = MENU_MULTI;
    menu.menu_type = MENU_OPTION;
    menu.option_tag = tag;
    options.push_back(menu);
    return *this;
}

Menu& Menu::Back(std::string text) {
    Menu menu = Menu(text);
    menu.menu_type = MENU_BACK;
    options.push_back(menu);
    return *this;
}

Menu Menu::CreateQuitMenu() {
    Menu menu = Menu("退出（或按ESC）")
        .Text("你想退出吗？")
        .Text("[YES] 如果想退出，请按 Y，SPACE 或 ENTER。")
        .Text("[NO] 想继续菜单，请按任何其他键。");
    menu.menu_type = MENU_QUIT;
    return menu;
}

Menu& Menu::Quit() {
    options.push_back(CreateQuitMenu());
    return *this;
}

Menu& Menu::Confirmation() {
    menu_type = MENU_CONFIRM;
    return *this;
}

Menu& Menu::get_selected_menu() {
    Menu& selected_menu = options[cursor];
    return selected_menu;
}

Menu& Menu::Acceptance(std::function<void()> cb) {
    callback = cb;
    return *this;
}