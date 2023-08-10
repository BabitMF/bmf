package main

import "C"
import (
        "encoding/json"
        "errors"
        "fmt"

        "github.com/babitmf/bmf-gosdk/bmf"
)

type PassThroughModuleOption struct {
        Value int32
}

type PassThroughModule struct {
        nodeId int32
        option PassThroughModuleOption
}

func (self *PassThroughModule) Process(task *bmf.Task) error {
        fmt.Println("Go-PassThrough process-in")
        defer fmt.Println("Go-PassThrough process-out")
        iids := task.GetInputStreamIds()
        oids := task.GetOutputStreamIds()

        gotEof := false
        for i, iid := range iids {
                for pkt, err := task.PopPacketFromInputQueue(iid); err == nil; {
                        defer pkt.Free()
                        if ok := task.FillOutputPacket(oids[i], pkt); !ok {
                                return errors.New("Fill output queue failed")
                        }

                        if pkt.Timestamp() == bmf.EOF {
                                gotEof = true
                        }

                        pkt, err = task.PopPacketFromInputQueue(iid)
                }
        }

        if gotEof {
                task.SetTimestamp(bmf.DONE)
        }
        return nil
}

func (self *PassThroughModule) Init() error {
        return nil
}

func (self *PassThroughModule) Reset() error {
        return errors.New("Reset is not supported")
}

func (self *PassThroughModule) Close() error {
        return nil
}

func (self *PassThroughModule) GetModuleInfo() (interface{}, error) {
        info := map[string]string{
                "NodeId": fmt.Sprintf("%d", self.nodeId),
        }

        return info, nil
}

func (self *PassThroughModule) NeedHungryCheck(istreamId int32) (bool, error) {
        return true, nil
}

func (self *PassThroughModule) IsHungry(istreamId int32) (bool, error) {
        return true, nil
}

func (self *PassThroughModule) IsInfinity() (bool, error) {
        return true, nil
}

func NewPassThroughModule(nodeId int32, option []byte) (bmf.Module, error) {
        m := &PassThroughModule{}
        err := json.Unmarshal(option, &m.option)
        if err != nil {
                return nil, err
        }
        m.nodeId = nodeId

        return m, nil
}

func RegisterPassThroughInfo(info bmf.ModuleInfo) {
        info.SetModuleDescription("Go PassThrough description")
        tag := bmf.NewModuleTag(bmf.BMF_TAG_UTILS|bmf.BMF_TAG_VIDEO_PROCESSOR)
        info.SetModuleTag(tag)
}

//export ConstructorRegister
func ConstructorRegister() {
        bmf.RegisterModuleConstructor("test_go_module", NewPassThroughModule, RegisterPassThroughInfo)
}

func main() {}
