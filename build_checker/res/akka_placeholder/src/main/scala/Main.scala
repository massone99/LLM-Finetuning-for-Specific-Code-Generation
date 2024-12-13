import akka.actor.{ActorSystem, Props}
import akka.actor.FSM
import scala.concurrent.duration._

// Define states
sealed trait LightState
case object Red extends LightState
case object Green extends LightState
case object Yellow extends LightState

// Define data
sealed trait LightData
case object Uninitialized extends LightData

// Define the TrafficLightActor
class TrafficLightActor extends FSM[LightState, LightData] {
  import context.dispatcher

  // Define state durations
  val redDuration: FiniteDuration = 3.seconds
  val greenDuration: FiniteDuration = 2.seconds
  val yellowDuration: FiniteDuration = 1.second

  startWith(Red, Uninitialized, stateTimeout = redDuration)

  when(Red, stateTimeout = redDuration) {
    case Event(StateTimeout, _) =>
      println("State: Red")
      goto(Green) using Uninitialized forMax greenDuration
  }

  when(Green, stateTimeout = greenDuration) {
    case Event(StateTimeout, _) =>
      println("State: Green")
      goto(Yellow) using Uninitialized forMax yellowDuration
  }

  when(Yellow, stateTimeout = yellowDuration) {
    case Event(StateTimeout, _) =>
      println("State: Yellow")
      goto(Red) using Uninitialized forMax redDuration
  }

  initialize()
}

// Usage Example (for testing purposes)
object TrafficLightApp extends App {
  val system = ActorSystem("TrafficLightSystem")
  val trafficLight = system.actorOf(Props[TrafficLightActor], "trafficLightActor")

  // Let the traffic light run for a few cycles before shutting down
  Thread.sleep(12000)
  system.terminate()
}